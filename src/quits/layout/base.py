"""Base types for qubit-to-coordinate layouts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..qldpc_code import QldpcCode

Coordinate = tuple[float, float]


@dataclass(frozen=True)
class LayoutMapping:
    data: dict[int, Coordinate]
    zcheck: dict[int, Coordinate]
    xcheck: dict[int, Coordinate]


class Layout(ABC):
    def __init__(self, code: "QldpcCode"):
        self.code = code

    @abstractmethod
    def mapping(self) -> LayoutMapping:
        """Return role-local coordinate maps for data and check qubits."""

    def data_positions(self) -> dict[int, Coordinate]:
        return dict(self.mapping().data)

    def zcheck_positions(self) -> dict[int, Coordinate]:
        return dict(self.mapping().zcheck)

    def xcheck_positions(self) -> dict[int, Coordinate]:
        return dict(self.mapping().xcheck)

    def node_positions(self, *, data_qubits, zcheck_qubits, xcheck_qubits) -> dict[int, Coordinate]:
        mapping = self.mapping()
        positions: dict[int, Coordinate] = {}
        positions.update(self._node_positions_for_role(mapping.data, data_qubits, "data"))
        positions.update(self._node_positions_for_role(mapping.zcheck, zcheck_qubits, "z-check"))
        positions.update(self._node_positions_for_role(mapping.xcheck, xcheck_qubits, "x-check"))
        return positions

    @staticmethod
    def _node_positions_for_role(role_positions, qubits, role_name: str) -> dict[int, Coordinate]:
        qubits = list(qubits)
        missing = [idx for idx in range(len(qubits)) if idx not in role_positions]
        if missing:
            raise ValueError(f"Layout mapping for {role_name} qubits is missing indices: {missing[:5]}")
        return {int(qubits[idx]): role_positions[idx] for idx in range(len(qubits))}


__all__ = ["Coordinate", "Layout", "LayoutMapping"]
