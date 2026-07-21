"""Batched GPU/CPU evaluation of DNNF circuits with PyTorch.

A smooth d-DNNF is an arithmetic circuit, so semiring evaluation can be
expressed as a layered tensor program:

1. Nodes are renumbered by depth (longest path from a leaf); each depth is
   one layer, and within a layer AND nodes precede OR nodes.
2. A layer is evaluated by gathering child values from the already-computed
   prefix (``index_select``) and segment-reducing them into parents
   (``scatter_add`` for AND, ``scatter_reduce(amin)`` or a numerically
   stable segment log-sum-exp for OR).
3. A batch dimension carries many literal-weight vectors — e.g. many
   evidence sets — through one forward pass.

Two semirings are provided:

* ``"logprob"``: values are log-weights; the root is ``log WMC``.
  Because the circuit is evaluated with autograd-friendly ops, calling
  ``.marginals(w)`` computes **all** posterior literal marginals in one
  backward pass — Darwiche's differential semantics of d-DNNF
  (``d logZ / d log w_l  =  P(l | evidence)``).
* ``"neglog"``: values are additive costs (neg-log probabilities); the root
  is the MPE cost, and ``.mpe(w)`` decodes the minimizing assignments.

Import this module only when torch is installed (``pip install
dnnf-lib[torch]``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .circuit import AND, FALSE, LIT, OR, TRUE, Circuit, lit_index


class TorchCircuit:
    """A circuit lowered to a layered tensor program.

    Parameters
    ----------
    circuit:
        A smooth d-DNNF (use ``compile_cnf(..., smooth=True)``).  Smoothness
        and determinism are the caller's responsibility for ``"logprob"``;
        ``"neglog"`` needs only decomposability.
    semiring:
        ``"logprob"`` or ``"neglog"``.
    device, dtype:
        Passed through to the index/constant tensors.
    """

    def __init__(
        self,
        circuit: Circuit,
        semiring: str = "logprob",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        if semiring not in ("logprob", "neglog"):
            raise ValueError("semiring must be 'logprob' or 'neglog'")
        self.circuit = circuit
        self.semiring = semiring
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_vars = circuit.num_vars
        self._build(circuit)

    # ------------------------------------------------------------------
    def _build(self, circuit: Circuit) -> None:
        n = len(circuit)
        depth = [0] * n
        for i in range(n):
            ch = circuit.children[i]
            depth[i] = 1 + max((depth[c] for c in ch), default=-1)

        # Renumber: sort by (depth, OR-after-AND, original index).  Leaves
        # (depth 0) are ordered literals first, then constants, so the leaf
        # layer can be assembled as [gathered lit weights | constants].
        def sort_key(i: int) -> Tuple[int, int, int]:
            kind = circuit.kinds[i]
            if depth[i] == 0:
                group = 0 if kind == LIT else 1
            else:
                group = 0 if kind == AND else 1
            return (depth[i], group, i)

        order = sorted(range(n), key=sort_key)
        new_pos = [0] * n
        for pos, old in enumerate(order):
            new_pos[old] = pos
        self.root_pos = new_pos[circuit.root]

        # Leaf layer.
        leaf_lit_idx: List[int] = []
        const_vals: List[float] = []
        zero = -math.inf if self.semiring == "logprob" else math.inf
        one = 0.0
        for old in order:
            if depth[old] != 0:
                break
            kind = circuit.kinds[old]
            if kind == LIT:
                leaf_lit_idx.append(lit_index(circuit.lits[old]))
            elif kind == TRUE:
                const_vals.append(one)
            else:  # FALSE
                const_vals.append(zero)
        self.num_leaves = len(leaf_lit_idx) + len(const_vals)
        self.leaf_lit_idx = torch.tensor(
            leaf_lit_idx, dtype=torch.long, device=self.device
        )
        self.const_vals = torch.tensor(
            const_vals, dtype=self.dtype, device=self.device
        )

        # Internal layers: per depth, edge lists for AND and OR segments.
        max_depth = max(depth) if n else 0
        self.layers: List[Dict[str, torch.Tensor | int]] = []
        for d in range(1, max_depth + 1):
            nodes = [old for old in order if depth[old] == d]
            and_child: List[int] = []
            and_seg: List[int] = []
            or_child: List[int] = []
            or_seg: List[int] = []
            n_and = 0
            n_or = 0
            for old in nodes:
                kids = [new_pos[c] for c in circuit.children[old]]
                if circuit.kinds[old] == AND:
                    and_child.extend(kids)
                    and_seg.extend([n_and] * len(kids))
                    n_and += 1
                else:
                    or_child.extend(kids)
                    or_seg.extend([n_or] * len(kids))
                    n_or += 1
            t = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
            self.layers.append(
                {
                    "and_child": t(and_child),
                    "and_seg": t(and_seg),
                    "or_child": t(or_child),
                    "or_seg": t(or_seg),
                    "n_and": n_and,
                    "n_or": n_or,
                }
            )

    # ------------------------------------------------------------------
    def _segment_or(
        self, gathered: torch.Tensor, seg: torch.Tensor, n_or: int
    ) -> torch.Tensor:
        B = gathered.shape[0]
        seg2 = seg.unsqueeze(0).expand(B, -1)
        if self.semiring == "neglog":
            out = torch.full(
                (B, n_or), math.inf, dtype=self.dtype, device=gathered.device
            )
            return out.scatter_reduce(
                1, seg2, gathered, reduce="amin", include_self=True
            )
        # logprob: numerically stable segment log-sum-exp.
        m = torch.full(
            (B, n_or), -math.inf, dtype=self.dtype, device=gathered.device
        )
        m = m.scatter_reduce(1, seg2, gathered, reduce="amax", include_self=True)
        m_det = m.detach()
        m_per_edge = m_det.gather(1, seg2)
        finite = m_per_edge > -math.inf
        shifted = torch.where(
            finite, gathered - m_per_edge, torch.zeros_like(gathered)
        )
        expd = torch.where(finite, shifted.exp(), torch.zeros_like(shifted))
        s = torch.zeros(B, n_or, dtype=self.dtype, device=gathered.device)
        s = s.scatter_add(1, seg2, expd)
        return torch.where(
            m_det > -math.inf,
            m_det + torch.log(s.clamp_min(1e-300)),
            torch.full_like(m_det, -math.inf),
        )

    def forward(
        self, weights: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor:
        """Evaluate the circuit.

        ``weights``: tensor of shape ``(B, 2 * num_vars)`` (or ``(2n,)``,
        auto-promoted), indexed by :func:`dnnf.circuit.lit_index` — log
        literal weights for ``"logprob"``, additive costs for ``"neglog"``.

        Returns the root value per batch element, shape ``(B,)`` — or, with
        ``return_all=True``, the full node-value tensor ``(B, num_nodes)``
        in layered order.
        """
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        B = weights.shape[0]
        needs_grad = weights.requires_grad and torch.is_grad_enabled()

        lit_vals = weights.index_select(1, self.leaf_lit_idx)
        consts = self.const_vals.unsqueeze(0).expand(B, -1)

        if not needs_grad:
            # Fast path: write layers into one preallocated buffer.  Layer
            # outputs occupy contiguous slices because nodes are numbered
            # by (depth, AND-before-OR).
            total = self.num_leaves + sum(
                layer["n_and"] + layer["n_or"] for layer in self.layers
            )
            acc = torch.empty(B, total, dtype=self.dtype, device=weights.device)
            acc[:, : lit_vals.shape[1]] = lit_vals
            acc[:, lit_vals.shape[1] : self.num_leaves] = consts
            offset = self.num_leaves
            for layer in self.layers:
                if layer["n_and"]:
                    gathered = acc.index_select(1, layer["and_child"])
                    seg2 = layer["and_seg"].unsqueeze(0).expand(B, -1)
                    s = torch.zeros(
                        B, layer["n_and"], dtype=self.dtype, device=acc.device
                    )
                    s.scatter_add_(1, seg2, gathered)
                    acc[:, offset : offset + layer["n_and"]] = s
                    offset += layer["n_and"]
                if layer["n_or"]:
                    gathered = acc.index_select(1, layer["or_child"])
                    acc[:, offset : offset + layer["n_or"]] = self._segment_or(
                        gathered, layer["or_seg"], layer["n_or"]
                    )
                    offset += layer["n_or"]
            if return_all:
                return acc
            return acc[:, self.root_pos]

        # Autograd path: functional (no in-place writes into the buffer).
        acc = torch.cat([lit_vals, consts], dim=1)
        for layer in self.layers:
            outs = []
            if layer["n_and"]:
                gathered = acc.index_select(1, layer["and_child"])
                seg2 = layer["and_seg"].unsqueeze(0).expand(B, -1)
                s = torch.zeros(
                    B, layer["n_and"], dtype=self.dtype, device=acc.device
                )
                outs.append(s.scatter_add(1, seg2, gathered))
            if layer["n_or"]:
                gathered = acc.index_select(1, layer["or_child"])
                outs.append(
                    self._segment_or(gathered, layer["or_seg"], layer["n_or"])
                )
            acc = torch.cat([acc] + outs, dim=1)
        if return_all:
            return acc
        return acc[:, self.root_pos]

    __call__ = forward

    # ------------------------------------------------------------------
    def marginals(self, weights: torch.Tensor) -> torch.Tensor:
        """Posterior literal marginals, ``"logprob"`` semiring only.

        For each batch row, returns a ``(B, 2n)`` tensor whose entry for
        literal ``l`` is ``P(l | theory, weights)``.  Computed as the
        gradient of log WMC with respect to the log literal weights — one
        forward + one backward pass for all marginals.
        """
        if self.semiring != "logprob":
            raise ValueError("marginals require the 'logprob' semiring")
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        w = weights.detach().clone().requires_grad_(True)
        log_z = self.forward(w)
        log_z.sum().backward()
        return w.grad.detach()

    def mpe(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Optional[Dict[int, bool]]]]:
        """Batched MPE, ``"neglog"`` semiring only.

        Returns ``(costs, assignments)``: per-row minimal model cost and the
        decoded assignment (None where the row is unsatisfiable).  The
        forward pass is batched tensor work; decoding is a linear top-down
        sweep per row.
        """
        if self.semiring != "neglog":
            raise ValueError("mpe requires the 'neglog' semiring")
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        vals = self.forward(weights, return_all=True).detach()
        costs = vals[:, self.root_pos]
        circuit = self.circuit
        n = len(circuit)
        depth = [0] * n
        for i in range(n):
            depth[i] = 1 + max(
                (depth[c] for c in circuit.children[i]), default=-1
            )

        def sort_key(i: int) -> Tuple[int, int, int]:
            kind = circuit.kinds[i]
            group = (
                (0 if kind == LIT else 1)
                if depth[i] == 0
                else (0 if kind == AND else 1)
            )
            return (depth[i], group, i)

        order = sorted(range(n), key=sort_key)
        new_pos = [0] * n
        for pos, old in enumerate(order):
            new_pos[old] = pos

        assignments: List[Optional[Dict[int, bool]]] = []
        vals_cpu = vals.cpu()
        for b in range(vals.shape[0]):
            if not torch.isfinite(costs[b]):
                assignments.append(None)
                continue
            row = vals_cpu[b]
            out: Dict[int, bool] = {}
            stack = [circuit.root]
            while stack:
                i = stack.pop()
                kind = circuit.kinds[i]
                if kind == LIT:
                    lit = circuit.lits[i]
                    out[abs(lit)] = lit > 0
                elif kind == AND:
                    stack.extend(circuit.children[i])
                elif kind == OR:
                    stack.append(
                        min(
                            circuit.children[i],
                            key=lambda c: row[new_pos[c]].item(),
                        )
                    )
            assignments.append(out)
        return costs, assignments

    # ------------------------------------------------------------------
    def weights_from_probs(
        self, probs: Dict[int, float], batch: int = 1
    ) -> torch.Tensor:
        """Build a ``(batch, 2n)`` weight tensor from per-variable
        ``P(v = true)`` (default 0.5), in this circuit's semiring."""
        row = []
        for v in range(1, self.num_vars + 1):
            p = probs.get(v, 0.5)
            for q in (p, 1.0 - p):
                if self.semiring == "logprob":
                    row.append(-math.inf if q <= 0 else math.log(q))
                else:
                    row.append(math.inf if q <= 0 else -math.log(q))
        t = torch.tensor(row, dtype=self.dtype, device=self.device)
        return t.unsqueeze(0).repeat(batch, 1)

    def condition(
        self, weights: torch.Tensor, evidence: Dict[int, bool]
    ) -> torch.Tensor:
        """Return a copy of ``weights`` with evidence-contradicting literals
        annihilated (``-inf`` for logprob, ``+inf`` for neglog)."""
        w = weights.clone()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        bad = -math.inf if self.semiring == "logprob" else math.inf
        for var, value in evidence.items():
            forbidden = -var if value else var
            w[:, lit_index(forbidden)] = bad
        return w
