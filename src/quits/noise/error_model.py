from dataclasses import dataclass
from numbers import Real
from typing import Tuple, Union


def _is_real_sequence(value, expected_len):
    """Check whether a value is a list/tuple of real numbers with a fixed length.

    :param value: Candidate object to validate as a numeric sequence.
    :param expected_len: Required number of entries in the sequence.
    :return: True if value is a sequence of real numbers with expected_len items.
    """

    if not isinstance(value, (tuple, list)):
        return False
    if len(value) != expected_len:
        return False
    return all(isinstance(p, Real) for p in value)


@dataclass(frozen=True)
class ErrorModel:
    """Noise model used during circuit construction.

    :param idle_error: Idle error channel; either a scalar rate or (px, py, pz).
    :param sqgate_error: Single-qubit gate error; either a scalar rate or (px, py, pz).
    :param tqgate_error: Two-qubit gate error; either a scalar rate or 15-entry Pauli channel tuple
        ordered as (IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ).
    :param spam_error: State-preparation-and-measurement (SPAM) error rate.
    """

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
        """Validate scalar or 3-entry Pauli-1 channel input.

        :param name: Field name used in error messages.
        :param value: Field value to validate.
        :return: None. Raises TypeError on invalid input.
        """

        if isinstance(value, Real):
            return
        if _is_real_sequence(value, 3):
            return
        raise TypeError(f"{name} must be a real number or length-3 tuple/list of real numbers.")

    @staticmethod
    def _validate_single_or_pauli2(name, value):
        """Validate scalar or 15-entry Pauli-2 channel input.

        :param name: Field name used in error messages.
        :param value: Field value to validate.
        :return: None. Raises TypeError on invalid input.
        """

        if isinstance(value, Real):
            return
        if _is_real_sequence(value, 15):
            return
        raise TypeError(f"{name} must be a real number or length-15 tuple/list of real numbers.")

    @classmethod
    def zero(cls):
        """Construct an all-zero noise model.

        :param cls: ErrorModel class used for construction.
        :return: ErrorModel with all rates/channels set to zero.
        """

        return cls()
