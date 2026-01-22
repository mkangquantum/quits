"""Public API facade for QUITS."""

from .circuit import get_qldpc_mem_circuit
from .decoder import (
    detector_error_model_to_matrix,
    sliding_window_bplsd_circuit_mem,
    sliding_window_bplsd_phenom_mem,
    sliding_window_bposd_circuit_mem,
    sliding_window_bposd_phenom_mem,
    sliding_window_circuit_mem,
    sliding_window_phenom_mem,
)
from .qldpc_code import BpcCode, HgpCode, QldpcCode, QlpCode, QlpCode2
from .simulation import get_stim_mem_result

__all__ = [
    "BpcCode",
    "HgpCode",
    "QldpcCode",
    "QlpCode",
    "QlpCode2",
    "get_qldpc_mem_circuit",
    "get_stim_mem_result",
    "detector_error_model_to_matrix",
    "sliding_window_phenom_mem",
    "sliding_window_bposd_phenom_mem",
    "sliding_window_bplsd_phenom_mem",
    "sliding_window_circuit_mem",
    "sliding_window_bposd_circuit_mem",
    "sliding_window_bplsd_circuit_mem",
]
