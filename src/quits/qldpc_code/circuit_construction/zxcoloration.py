from .base import CircuitBuilder


class ZXColorationBuilder(CircuitBuilder):
    name = "zxcoloration"

    def build(self, code, **opts):
        raise NotImplementedError("ZX coloration circuit construction is not implemented yet.")
