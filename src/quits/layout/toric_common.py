"""Shared toric embedding machinery for supported code families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import lcm

from .base import Layout, LayoutMapping


class AbstractToricLayout(Layout, ABC):
    """Generic toric embedding over a regular permutation action."""

    def __init__(self, code):
        super().__init__(code)
        self._left_size = int(self._role_size())
        self._a_terms = tuple(self._normalize_permutation(perm) for perm in self._a_term_permutations())
        self._b_terms = tuple(self._normalize_permutation(perm) for perm in self._b_term_permutations())
        if len(self._a_terms) != 3 or len(self._b_terms) != 3:
            raise ValueError(self._toric_error_message())
        self._selection = self._select_generators()
        self._mapping = self._build_mapping()

    def mapping(self) -> LayoutMapping:
        return self._mapping

    @property
    def torus_shape(self) -> tuple[int, int]:
        return (self._selection["order_b"], self._selection["order_a"])

    @abstractmethod
    def _role_size(self) -> int:
        """Return the number of left data qubits / X checks / Z checks."""

    @abstractmethod
    def _a_term_permutations(self) -> tuple[tuple[int, ...], ...]:
        """Return the three A-term permutations acting on role-local indices."""

    @abstractmethod
    def _b_term_permutations(self) -> tuple[tuple[int, ...], ...]:
        """Return the three B-term permutations acting on role-local indices."""

    @abstractmethod
    def _toric_error_message(self) -> str:
        """Return the family-specific error for unsupported instances."""

    def _select_generators(self):
        best = None
        for i, j in self._ordered_pairs():
            inv_a_j = self._perm_inv(self._a_terms[j])
            delta_a = self._perm_comp(self._a_terms[i], inv_a_j)
            order_a = self._perm_order(delta_a)
            if order_a == 0:
                continue
            for g, h in self._ordered_pairs():
                inv_b_h = self._perm_inv(self._b_terms[h])
                delta_b = self._perm_comp(self._b_terms[g], inv_b_h)
                order_b = self._perm_order(delta_b)
                if order_b == 0 or order_a * order_b != self._left_size:
                    continue
                alpha_coords = self._alpha_coordinates(delta_b, order_b, delta_a, order_a)
                if alpha_coords is None:
                    continue
                score = self._pair_score(i, j, g, h)
                choice = {
                    "i": i,
                    "j": j,
                    "g": g,
                    "h": h,
                    "inv_a_j": inv_a_j,
                    "b_g": self._b_terms[g],
                    "order_a": order_a,
                    "order_b": order_b,
                    "alpha_coords": alpha_coords,
                }
                if best is None or score < best[0]:
                    best = (score, choice)

        if best is None:
            raise ValueError(self._toric_error_message())
        return best[1]

    def _build_mapping(self) -> LayoutMapping:
        data = {}
        zcheck = {}
        xcheck = {}
        half = self._left_size
        inv_a_j = self._selection["inv_a_j"]
        b_g = self._selection["b_g"]
        right_perm = self._perm_comp(inv_a_j, b_g)

        for alpha, (coord_x, coord_y) in self._selection["alpha_coords"].items():
            left_index = alpha
            z_index = b_g[alpha]
            x_index = inv_a_j[alpha]
            right_index = right_perm[alpha]

            data[left_index] = (2 * coord_x, 2 * coord_y)
            data[half + right_index] = (2 * coord_x + 1, 2 * coord_y + 1)
            zcheck[z_index] = (2 * coord_x + 1, 2 * coord_y)
            xcheck[x_index] = (2 * coord_x, 2 * coord_y + 1)

        return LayoutMapping(data=data, zcheck=zcheck, xcheck=xcheck)

    def _alpha_coordinates(self, delta_b, order_b, delta_a, order_a):
        coords = {}
        origin = 0
        for coord_x in range(order_b):
            horiz = self._perm_pow(delta_b, coord_x)
            for coord_y in range(order_a):
                alpha = self._perm_pow(delta_a, coord_y)[horiz[origin]]
                if alpha in coords:
                    return None
                coords[alpha] = (coord_x, coord_y)
        if len(coords) != self._left_size:
            return None
        return coords

    def _normalize_permutation(self, perm) -> tuple[int, ...]:
        perm = tuple(int(value) for value in perm)
        if len(perm) != self._left_size or set(perm) != set(range(self._left_size)):
            raise ValueError(self._toric_error_message())
        return perm

    @staticmethod
    def _ordered_pairs():
        return ((2, 1), (1, 2), (0, 2), (0, 1), (2, 0), (1, 0))

    @staticmethod
    def _pair_score(i, j, g, h):
        pair_positions = {(2, 1): 0, (1, 2): 1, (0, 2): 2, (0, 1): 3, (2, 0): 4, (1, 0): 5}
        return (pair_positions[(i, j)], pair_positions[(g, h)], i, j, g, h)

    @staticmethod
    def _perm_comp(lhs, rhs):
        return tuple(lhs[rhs[idx]] for idx in range(len(lhs)))

    @staticmethod
    def _perm_inv(perm):
        inv = [0] * len(perm)
        for idx, value in enumerate(perm):
            inv[value] = idx
        return tuple(inv)

    @staticmethod
    def _perm_pow(perm, exponent: int):
        result = tuple(range(len(perm)))
        base = perm
        while exponent:
            if exponent & 1:
                result = AbstractToricLayout._perm_comp(base, result)
            base = AbstractToricLayout._perm_comp(base, base)
            exponent //= 2
        return result

    @staticmethod
    def _perm_order(perm):
        if perm == tuple(range(len(perm))):
            return 1
        seen = [False] * len(perm)
        order = 1
        for idx in range(len(perm)):
            if seen[idx]:
                continue
            cycle_length = 0
            cur = idx
            while not seen[cur]:
                seen[cur] = True
                cur = perm[cur]
                cycle_length += 1
            order = lcm(order, cycle_length)
        return order


__all__ = ["AbstractToricLayout"]
