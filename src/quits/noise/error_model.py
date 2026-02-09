from dataclasses import dataclass
from numbers import Real
from typing import Tuple, Union


def _is_real_sequence(value, expected_len):
    if not isinstance(value, (tuple, list)):
        return False
    if len(value) != expected_len:
        return False
    return all(isinstance(p, Real) for p in value)


@dataclass(frozen=True)
class ErrorModel:
    idle_error: Union[float, Tuple[float, float, float]] = 0.0
    sqgate_error: Union[float, Tuple[float, float, float]] = 0.0
    tqgate_error: Union[float, Tuple[float, ...]] = 0.0
    spam_error: float = 0.0

    def __post_init__(self):
        self._validate_single_or_pauli1("idle_error", self.idle_error)
        self._validate_single_or_pauli1("sqgate_error", self.sqgate_error)
        self._validate_single_or_pauli2("tqgate_error", self.tqgate_error)
        if not isinstance(self.spam_error, Real):
            raise TypeError("spam_error must be a real number.")

    @staticmethod
    def _validate_single_or_pauli1(name, value):
        if isinstance(value, Real):
            return
        if _is_real_sequence(value, 3):
            return
        raise TypeError(f"{name} must be a real number or length-3 tuple/list of real numbers.")

    @staticmethod
    def _validate_single_or_pauli2(name, value):
        if isinstance(value, Real):
            return
        if _is_real_sequence(value, 15):
            return
        raise TypeError(f"{name} must be a real number or length-15 tuple/list of real numbers.")

    @classmethod
    def zero(cls):
        return cls()
