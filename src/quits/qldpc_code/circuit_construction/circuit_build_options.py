from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitBuildOptions:
    """Configuration options for circuit-construction behavior.

    :param get_all_detectors: Whether to include detectors from both X and Z bases.
    :param noisy_zeroth_round: Whether the zeroth-round initialization is noisy.
    :param noisy_final_meas: Whether the final data-qubit measurement is noisy.
    """

    get_all_detectors: bool = False
    noisy_zeroth_round: bool = True
    noisy_final_meas: bool = False

    def __post_init__(self):
        if not isinstance(self.get_all_detectors, bool):
            raise TypeError("get_all_detectors must be a bool.")
        if not isinstance(self.noisy_zeroth_round, bool):
            raise TypeError("noisy_zeroth_round must be a bool.")
        if not isinstance(self.noisy_final_meas, bool):
            raise TypeError("noisy_final_meas must be a bool.")
