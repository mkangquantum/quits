from .base import CircuitBuilder


class FreeformBuilder(CircuitBuilder):
    name = "freeform"

    def build(self, code, **opts):
        raise NotImplementedError("Freeform circuit construction is not implemented yet.")
