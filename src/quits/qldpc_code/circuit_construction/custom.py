from .base import CircuitBuilder


class CustomBuilder(CircuitBuilder):
    name = "custom"

    def build(self, code, **opts):
        raise NotImplementedError("Custom circuit construction is not implemented yet.")
