"""Row-based transversal Tanner-graph layouts for generic CSS codes."""

from __future__ import annotations

from math import ceil

from .base import Layout, LayoutMapping


class TransversalLayout(Layout):
    def __init__(
        self,
        code,
        *,
        center_checks: bool = True,
        data_rows: int = 1,
        zcheck_rows: int = 1,
        xcheck_rows: int = 1,
    ):
        super().__init__(code)
        if code.hz is None or code.hx is None:
            raise ValueError("TransversalLayout requires code.hz and code.hx to be set.")
        if code.hz.shape[1] != code.hx.shape[1]:
            raise ValueError("TransversalLayout requires code.hz and code.hx to have the same number of columns.")
        self.center_checks = center_checks
        self.data_rows = self._validate_rows(data_rows, "data_rows")
        self.zcheck_rows = self._validate_rows(zcheck_rows, "zcheck_rows")
        self.xcheck_rows = self._validate_rows(xcheck_rows, "xcheck_rows")
        self._mapping = self._build_mapping()

    def mapping(self) -> LayoutMapping:
        return self._mapping

    def _build_mapping(self) -> LayoutMapping:
        n_data = int(self.code.hz.shape[1])
        n_z = int(self.code.hz.shape[0])
        n_x = int(self.code.hx.shape[0])

        data_cols = self._num_columns(n_data, self.data_rows)
        z_cols = self._num_columns(n_z, self.zcheck_rows)
        x_cols = self._num_columns(n_x, self.xcheck_rows)

        data = self._role_positions(
            count=n_data,
            rows=self.data_rows,
            x_offset=0.0,
            y_start=0.0,
            y_step=-1.0,
        )
        if self.center_checks:
            data_center_x = 0.5 * (data_cols - 1) if data_cols > 0 else 0.0
            z_offset_x = data_center_x - (0.5 * (z_cols - 1) if z_cols > 0 else 0.0)
            x_offset_x = data_center_x - (0.5 * (x_cols - 1) if x_cols > 0 else 0.0)
        else:
            z_offset_x = 0.0
            x_offset_x = 0.0

        zcheck = self._role_positions(
            count=n_z,
            rows=self.zcheck_rows,
            x_offset=z_offset_x - 0.5,
            y_start=-(float(self.data_rows) + 1.0),
            y_step=-1.0,
        )
        xcheck = self._role_positions(
            count=n_x,
            rows=self.xcheck_rows,
            x_offset=x_offset_x + 0.5,
            y_start=2.0,
            y_step=1.0,
        )
        return LayoutMapping(data=data, zcheck=zcheck, xcheck=xcheck)

    @staticmethod
    def _validate_rows(rows: int, name: str) -> int:
        if not isinstance(rows, int) or rows <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return rows

    @staticmethod
    def _num_columns(count: int, rows: int) -> int:
        if count == 0:
            return 0
        return int(ceil(count / rows))

    def _role_positions(self, *, count: int, rows: int, x_offset: float, y_start: float, y_step: float):
        cols = self._num_columns(count, rows)
        positions = {}
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            positions[idx] = (float(col + x_offset), float(y_start + row * y_step))
        return positions


__all__ = ["TransversalLayout"]
