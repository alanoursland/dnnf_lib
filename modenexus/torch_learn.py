"""Gradient-based prior learning through the differentiable circuit.

Where :meth:`modenexus.diagnosis.CompiledSystem.fit_priors` runs exact EM
(ideal for independent categorical priors and modest data),
:class:`PriorLearner` trains the same parameters by SGD on the torch
backend: per-variable logits, softmax-normalized into value weights,
maximizing the log-likelihood of partially observed telemetry via
batched masked log-WMC.  The gradient path scales to large telemetry
sets (observations dedup into weighted unique evidence masks and batch
through one layered forward), runs on GPU, and composes with anything
else differentiable — e.g. an observation model producing the masks.

Requires torch (``pip install modenexus[torch]``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from .torch_backend import TorchCircuit


class PriorLearner:
    """Trainable value priors for named variables of a compiled system.

    Parameters
    ----------
    system:
        A :class:`modenexus.diagnosis.CompiledSystem`.
    names:
        Variables whose priors to learn (default: the mode variables).
        Their current priors initialize the logits.
    """

    def __init__(
        self,
        system,
        names: Optional[Sequence[str]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.system = system
        self.names = list(system.mode_vars if names is None else names)
        self.tc = TorchCircuit(
            system.circuit, semiring="logprob", device=device, dtype=dtype
        )
        self.device = torch.device(device)
        self.dtype = dtype
        spec = system.circuit.spec
        total = spec.total

        # Fixed log-weights for non-trained slots.
        base = [
            -math.inf if w <= 0 else math.log(w) for w in system._weights
        ]
        self.base = torch.tensor(base, dtype=dtype, device=self.device)

        self.logits: List[torch.Tensor] = []
        slot_idx: List[int] = []
        trainable = torch.zeros(total, dtype=torch.bool, device=self.device)
        for name in self.names:
            var = system.vars[name]
            slots = [
                spec.mvlit(var.fd_var, i) for i in range(len(var.values))
            ]
            init = torch.tensor(
                [
                    math.log(max(system._weights[s], 1e-12))
                    for s in slots
                ],
                dtype=dtype,
                device=self.device,
            )
            self.logits.append(init.clone().requires_grad_(True))
            slot_idx.extend(slots)
            trainable[slots] = True
        self.slot_idx = torch.tensor(
            slot_idx, dtype=torch.long, device=self.device
        )
        self.trainable = trainable

    # ------------------------------------------------------------------
    def masks_for(self, observations: Sequence[Dict]) -> torch.Tensor:
        """Additive evidence masks, one row per observation: 0 where a
        value stays possible, -inf where the evidence rules it out."""
        spec = self.system.circuit.spec
        rows = torch.zeros(
            len(observations), spec.total, dtype=self.dtype,
            device=self.device,
        )
        for r, obs in enumerate(observations):
            for name, value in obs.items():
                var = self.system.vars[name]
                chosen = self.system._value_index(name, value)
                for i in range(len(var.values)):
                    if i != chosen:
                        rows[r, spec.mvlit(var.fd_var, i)] = -math.inf
        return rows

    def _weights(self, masks: torch.Tensor) -> torch.Tensor:
        parts = [torch.log_softmax(l, dim=0) for l in self.logits]
        padded = torch.zeros(
            self.base.shape[0], dtype=self.dtype, device=self.device
        ).index_add(0, self.slot_idx, torch.cat(parts))
        w = torch.where(self.trainable, padded, self.base)
        return w.unsqueeze(0) + masks

    def log_likelihood(self, masks: torch.Tensor) -> torch.Tensor:
        """Per-row ``log P(observation)`` under the current logits."""
        return self.tc(self._weights(masks))

    # ------------------------------------------------------------------
    def fit(
        self,
        observations: Sequence[Dict],
        epochs: int = 300,
        lr: float = 0.05,
        verbose: bool = False,
    ) -> List[float]:
        """Maximize telemetry log-likelihood by Adam; returns the average
        log-likelihood trace and writes the learned priors back into the
        compiled system (so subsequent queries use them).

        Identical observations are deduplicated into weighted unique
        evidence masks, so cost scales with distinct evidence patterns,
        not raw telemetry length.
        """
        groups: Dict[tuple, int] = {}
        for obs in observations:
            groups[tuple(sorted(obs.items()))] = (
                groups.get(tuple(sorted(obs.items())), 0) + 1
            )
        unique = [dict(k) for k in groups]
        counts = torch.tensor(
            [groups[tuple(sorted(u.items()))] for u in unique],
            dtype=self.dtype, device=self.device,
        )
        n = counts.sum()
        masks = self.masks_for(unique)

        opt = torch.optim.Adam(self.logits, lr=lr)
        history: List[float] = []
        for epoch in range(epochs):
            opt.zero_grad()
            ll = self.log_likelihood(masks)
            loss = -(counts * ll).sum() / n
            loss.backward()
            opt.step()
            history.append(-loss.item())
            if verbose and epoch % 50 == 0:  # pragma: no cover
                print(f"epoch {epoch}: avg log-lik {-loss.item():.6f}")
        self._write_back()
        return history

    def probs(self) -> Dict[str, Dict[str, float]]:
        """Current learned priors per variable."""
        out: Dict[str, Dict[str, float]] = {}
        for name, logits in zip(self.names, self.logits):
            var = self.system.vars[name]
            p = torch.softmax(logits.detach(), dim=0)
            out[name] = {
                value: p[i].item() for i, value in enumerate(var.values)
            }
        return out

    def _write_back(self) -> None:
        spec = self.system.circuit.spec
        for name, dist in self.probs().items():
            var = self.system.vars[name]
            for i, value in enumerate(var.values):
                w = dist[value]
                mvlit = spec.mvlit(var.fd_var, i)
                self.system._weights[mvlit] = w
                self.system._costs[mvlit] = (
                    math.inf if w <= 0 else -math.log(w)
                )


class ObservationTrainer:
    """Train a neural observation model end-to-end through the circuit.

    The network maps raw sensor input to per-value log-likelihoods for
    the named observables (Pearl virtual evidence); the training signal
    is the log-WMC of that soft evidence under the compiled model — so
    the detectors learn from telemetry plus the logical structure, with
    **no labels for the observables themselves**.

    The network's output width must be ``sum(len(var.values) for the
    named observables)``, in name order, each block in the variable's
    value order.
    """

    def __init__(self, system, names, device="cpu",
                 dtype=torch.float64):
        self.system = system
        self.names = list(names)
        self.tc = TorchCircuit(system.circuit, semiring="logprob",
                               device=device, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        spec = system.circuit.spec
        self.base = torch.tensor(
            [-math.inf if w <= 0 else math.log(w)
             for w in system._weights],
            dtype=dtype, device=self.device,
        )
        slots = []
        self.blocks = []
        for name in self.names:
            var = system.vars[name]
            start = len(slots)
            slots.extend(
                spec.mvlit(var.fd_var, i) for i in range(len(var.values))
            )
            self.blocks.append((start, len(slots)))
        self.slots = torch.tensor(slots, dtype=torch.long,
                                  device=self.device)
        self.width = len(slots)

    def log_likelihood(self, local: torch.Tensor,
                       masks: "torch.Tensor" = None) -> torch.Tensor:
        """Per-row log-WMC given (B, width) local log-likelihoods.

        Each observable's block is log-softmax-normalized first: virtual
        evidence is only defined up to scale, and without normalization
        the likelihood is unbounded (the network could inflate every
        value's likelihood at once)."""
        local = torch.cat(
            [
                torch.log_softmax(local[:, a:b], dim=1)
                for a, b in self.blocks
            ],
            dim=1,
        )
        B = local.shape[0]
        pad = torch.zeros(B, self.base.shape[0], dtype=self.dtype,
                          device=self.device)
        pad = pad.index_add(1, self.slots, local)
        w = self.base.unsqueeze(0) + pad
        if masks is not None:
            w = w + masks
        return self.tc(w)

    def masks_for(self, observations) -> torch.Tensor:
        """Hard-evidence masks (B, total): -inf on ruled-out values.
        This is the grounding signal — soft neural evidence alone is
        degenerate (reporting the a-priori likely value is optimal);
        hard evidence elsewhere in the structure is what forces the
        detectors to track their inputs."""
        spec = self.system.circuit.spec
        rows = torch.zeros(len(observations), spec.total,
                           dtype=self.dtype, device=self.device)
        for r, obs in enumerate(observations):
            for name, value in obs.items():
                var = self.system.vars[name]
                chosen = self.system._value_index(name, value)
                for i in range(len(var.values)):
                    if i != chosen:
                        rows[r, spec.mvlit(var.fd_var, i)] = -math.inf
        return rows

    def fit(self, net: "torch.nn.Module", inputs: torch.Tensor,
            masks: "torch.Tensor" = None,
            epochs: int = 200, lr: float = 0.01):
        """Maximize mean log-WMC of ``net(inputs)`` combined with the
        per-row hard-evidence ``masks``; returns the log-lik trace."""
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        history = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = -self.log_likelihood(
                net(inputs).to(self.dtype), masks
            ).mean()
            loss.backward()
            opt.step()
            history.append(-loss.item())
        return history
