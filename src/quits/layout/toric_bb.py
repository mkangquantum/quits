"""BB-specific toric embeddings."""

from __future__ import annotations

from ..qldpc_code.bb import BbCode
from .toric_common import AbstractToricLayout

_BB_TORIC_LAYOUT_ERROR = (
    "ToricLayout only supports BbCode instances with invertible local-shift gaps."
)


class BbToricLayout(AbstractToricLayout):
    def __init__(self, code):
        if not isinstance(code, BbCode):
            raise ValueError(_BB_TORIC_LAYOUT_ERROR)
        if (
            len(code.A_x_pows) != 1
            or len(code.A_y_pows) != 2
            or len(code.B_y_pows) != 1
            or len(code.B_x_pows) != 2
        ):
            raise ValueError(_BB_TORIC_LAYOUT_ERROR)
        self.l = int(code.l)
        self.m = int(code.m)
        self.u = int(code.A_x_pows[0])
        self.p, self.q = sorted(int(power) for power in code.A_y_pows)
        self.v = int(code.B_y_pows[0])
        self.r, self.s = sorted(int(power) for power in code.B_x_pows)
        super().__init__(code)

    def _role_size(self) -> int:
        return self.l * self.m

    def _a_term_permutations(self) -> tuple[tuple[int, ...], ...]:
        return (
            self._shift_perm(self.u, 0),
            self._shift_perm(0, self.p),
            self._shift_perm(0, self.q),
        )

    def _b_term_permutations(self) -> tuple[tuple[int, ...], ...]:
        return (
            self._shift_perm(0, self.v),
            self._shift_perm(self.r, 0),
            self._shift_perm(self.s, 0),
        )

    def _toric_error_message(self) -> str:
        return _BB_TORIC_LAYOUT_ERROR

    def _shift_perm(self, dx: int, dy: int) -> tuple[int, ...]:
        perm = []
        for ax in range(self.l):
            for by in range(self.m):
                perm.append(self._index((ax + dx) % self.l, (by + dy) % self.m))
        return tuple(perm)

    def _index(self, ax: int, by: int) -> int:
        return ax * self.m + by


__all__ = ["BbToricLayout"]
