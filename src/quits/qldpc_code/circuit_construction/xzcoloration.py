from .base import CircuitBuilder


class XZColorationBuilder(CircuitBuilder):
    name = "xzcoloration"

    def build(self, code, **opts):
        raise NotImplementedError("XZ coloration circuit construction is not implemented yet.")
