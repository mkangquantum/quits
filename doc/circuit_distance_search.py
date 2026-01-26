"""
Circuit-distance search experiments (arXiv:2504.02673).

This script builds a set of BPC codes, generates the corresponding memory
circuits, and searches for undetectable logical errors using Stim.

Warning:
    Some configurations are very memory intensive. For the BPC [[144,8,12]]
    example, we used ~100 GB RAM on a cluster. Use smaller search limits or
    set `dont_explore_edges_increasing_symptom_degree=True` for laptops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import stim

from quits.circuit import get_qldpc_mem_circuit
from quits.decoder import detector_error_model_to_matrix
from quits.qldpc_code import BpcCode


@dataclass(frozen=True)
class SearchConfig:
    label: str
    factor: int
    lift_size: int
    p1: Tuple[int, ...]
    p2: Tuple[int, ...]
    num_rounds: int
    basis: str = "Z"
    seed: int = 1


def _build_bpc_code(cfg: SearchConfig) -> BpcCode:
    code = BpcCode(list(cfg.p1), list(cfg.p2), cfg.lift_size, cfg.factor)
    code.build_circuit(seed=cfg.seed)
    return code


def _build_circuit(code: BpcCode, num_rounds: int, basis: str) -> stim.Circuit:
    # Physical error rate does not affect circuit-distance search.
    p = 2e-3
    return stim.Circuit(get_qldpc_mem_circuit(code, p, p, p, p, num_rounds, basis=basis))


def run_distance_search(
    cfg: SearchConfig,
    *,
    dont_explore_detection_event_sets_with_size_above: int = 6,
    dont_explore_edges_with_degree_above: int = 6,
    dont_explore_edges_increasing_symptom_degree: bool = False,
) -> int:
    code = _build_bpc_code(cfg)
    circuit = _build_circuit(code, cfg.num_rounds, cfg.basis)

    model = circuit.detector_error_model(decompose_errors=False)
    detector_error_model_to_matrix(model)

    err_list = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=(
            dont_explore_detection_event_sets_with_size_above
        ),
        dont_explore_edges_with_degree_above=dont_explore_edges_with_degree_above,
        dont_explore_edges_increasing_symptom_degree=(
            dont_explore_edges_increasing_symptom_degree
        ),
    )
    return len(err_list)


def iter_configs() -> Iterable[SearchConfig]:
    yield SearchConfig(
        label="[[72,8,8]]",
        factor=3,
        lift_size=12,
        p1=(0, 1, 5),
        p2=(0, 1, 8),
        num_rounds=2,
    )
    # yield SearchConfig(
    #     label="[[90,8,10]]",
    #     factor=3,
    #     lift_size=15,
    #     p1=(0, 1, 5),
    #     p2=(0, 2, 7),
    #     num_rounds=2,
    # )
    # yield SearchConfig(
    #     label="[[144,8,12]]",
    #     factor=3,
    #     lift_size=24,
    #     p1=(0, 1, 5),
    #     p2=(0, 1, 11),
    #     num_rounds=1,
    # )


def main() -> None:
    for cfg in iter_configs():
        count = run_distance_search(cfg)
        print(cfg.label, count)


if __name__ == "__main__":
    main()

