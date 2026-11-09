"""Split-aware stratified validation sampler with epoch cycling.

Each (split, topology) group maintains its own shuffled queue. Every call to
``__iter__`` pops *K_split* indices from each group.  Once a group's queue is
exhausted the group reshuffles and refills, so every row is visited exactly
once before any repetition within that group.

The per-call yield count is deterministic:
    total = sum over splits of  (num_topologies_in_split × K_split)
so ``batch_size`` can be set to that total with no tail-batch issue.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Iterator

from torch.utils.data import Sampler


class SplitStratifiedCycleSampler(Sampler[int]):
    """Cycle-based stratified sampler that respects dataset splits.

    Parameters
    ----------
    split_labels : list[str]
        Per-row split name (``"validation"`` / ``"test"`` / …).
    topology_labels : list[str]
        Per-row topology name.
    samples_per_split : dict[str, int]
        Mapping ``split_name → K`` (how many rows per topology per call).
        Splits not listed here fall back to ``default_k``.
    default_k : int
        Default samples-per-topology for unlisted splits.
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        split_labels: list[str],
        topology_labels: list[str],
        samples_per_split: dict[str, int],
        default_k: int = 3,
        seed: int = 42,
        fixed: bool = False,
    ):
        assert len(split_labels) == len(topology_labels)
        self.samples_per_split = samples_per_split
        self.default_k = default_k
        self.seed = seed
        self.fixed = fixed
        self._call_count = 0

        self._groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, (split, topo) in enumerate(zip(split_labels, topology_labels)):
            self._groups[(split, topo)].append(idx)

        self._queues: dict[tuple[str, str], deque[int]] = {}
        self._group_seeds: dict[tuple[str, str], int] = {}
        for i, key in enumerate(sorted(self._groups)):
            self._group_seeds[key] = seed + i * 1000
            self._queues[key] = deque()

        self._total = 0
        for (split, _topo), indices in self._groups.items():
            k = self.samples_per_split.get(split, self.default_k)
            self._total += min(k, len(indices))

        # Fixed mode: draw once at construction and reuse every call.
        self._fixed_indices: list[int] | None = None
        if fixed:
            self._fixed_indices = self._draw(call_index=1)

    def _refill(self, key: tuple[str, str]) -> None:
        rng = random.Random(self._group_seeds[key])
        self._group_seeds[key] += 1
        pool = list(self._groups[key])
        rng.shuffle(pool)
        self._queues[key] = deque(pool)

    def _draw(self, call_index: int) -> list[int]:
        rng = random.Random(self.seed + call_index)
        selected: list[int] = []
        for key in sorted(self._groups):
            split, _topo = key
            k = self.samples_per_split.get(split, self.default_k)
            k = min(k, len(self._groups[key]))
            q = self._queues[key]
            picked: list[int] = []
            for _ in range(k):
                if not q:
                    self._refill(key)
                    q = self._queues[key]
                picked.append(q.popleft())
            selected.extend(picked)
        rng.shuffle(selected)
        return selected

    def __iter__(self) -> Iterator[int]:
        if self._fixed_indices is not None:
            return iter(self._fixed_indices)
        self._call_count += 1
        return iter(self._draw(self._call_count))

    def __len__(self) -> int:
        return self._total

    def summary(self) -> dict:
        """Return a human-readable summary of group sizes and K values."""
        split_topo_counts: dict[str, dict[str, int]] = defaultdict(dict)
        for (split, topo), indices in sorted(self._groups.items()):
            split_topo_counts[split][topo] = len(indices)

        info: dict = {}
        for split in sorted(split_topo_counts):
            k = self.samples_per_split.get(split, self.default_k)
            n_topos = len(split_topo_counts[split])
            n_rows = sum(split_topo_counts[split].values())
            per_call = n_topos * min(k, min(split_topo_counts[split].values()))
            cycles = max(split_topo_counts[split].values()) // k if k > 0 else 0
            info[split] = {
                "topologies": n_topos,
                "total_rows": n_rows,
                "k_per_topology": k,
                "rows_per_call": per_call,
                "full_cycles_before_repeat": cycles,
            }
        info["total_per_call"] = self._total
        return info


def build_split_and_topology_labels(dataset) -> tuple[list[str], list[str]]:
    """Extract per-row (split, topology) from an RLHFDataset.

    Reads ``extra_info.split`` and ``extra_info.env_kwargs.circuit_name``.
    """
    splits: list[str] = []
    topos: list[str] = []
    dataframe = dataset.dataframe
    for idx in range(len(dataframe)):
        row = dataframe[idx]
        ei = row.get("extra_info", {})
        if isinstance(ei, str):
            import ast
            ei = ast.literal_eval(ei)
        if not isinstance(ei, dict):
            ei = {}
        split = str(ei.get("split", "unknown"))
        if split in ("val", "valid"):
            split = "validation"
        circuit = str(ei.get("env_kwargs", {}).get("circuit_name", "unknown"))
        splits.append(split)
        topos.append(circuit)
    return splits, topos
