"""Public API facade for QUITS."""

from .noise import ErrorModel
from .qldpc_code.circuit_construction import CircuitBuildOptions
from .qldpc_code.circuit_construction.cardinal import CardinalBuilder
from .decoder import (
    detector_error_model_to_matrix,
    sliding_window_bplsd_circuit_mem,
    sliding_window_bplsd_phenom_mem,
    sliding_window_bposd_circuit_mem,
    sliding_window_bposd_phenom_mem,
    sliding_window_circuit_mem,
    sliding_window_phenom_mem,
)
from .qldpc_code import BbCode, BpcCode, HgpCode, LcsCode, QldpcCode, QlpCode, QlpPolyCode
from .simulation import get_stim_mem_result

__all__ = [
    "BbCode",
    "BpcCode",
    "HgpCode",
    "QldpcCode",
    "QlpCode",
    "QlpPolyCode",
    "LcsCode",
    "ErrorModel",
    "CircuitBuildOptions",
    "get_cardinal_circuit",
    "get_stim_mem_result",
    "detector_error_model_to_matrix",
    "sliding_window_phenom_mem",
    "sliding_window_bposd_phenom_mem",
    "sliding_window_bplsd_phenom_mem",
    "sliding_window_circuit_mem",
    "sliding_window_bposd_circuit_mem",
    "sliding_window_bplsd_circuit_mem",
]


def get_cardinal_circuit(
    code,
    error_model=None,
    num_rounds=0,
    basis="Z",
    circuit_build_options=None,
):
    if error_model is None:
        error_model = ErrorModel()
    if circuit_build_options is None:
        circuit_build_options = CircuitBuildOptions()
    elif not isinstance(circuit_build_options, CircuitBuildOptions):
        raise TypeError("circuit_build_options must be a CircuitBuildOptions instance.")
    return CardinalBuilder(code).get_cardinal_circuit(
        error_model=error_model,
        num_rounds=num_rounds,
        basis=basis,
        circuit_build_options=circuit_build_options,
    )
