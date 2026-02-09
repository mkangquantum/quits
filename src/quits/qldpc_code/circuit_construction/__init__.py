from .cardinal import CardinalBuilder
from .circuit_build_options import CircuitBuildOptions
from .freeform import FreeformBuilder
from .zxcoloration import ZXColorationBuilder

_BUILDERS = {
    CardinalBuilder.name: CardinalBuilder,
    ZXColorationBuilder.name: ZXColorationBuilder,
    FreeformBuilder.name: FreeformBuilder,
}


def get_builder(name, code=None):
    if name is None:
        name = "cardinal"
    builder_cls = _BUILDERS.get(name)
    if builder_cls is None:
        raise ValueError(f"Unknown circuit construction strategy: {name}")
    if name == CardinalBuilder.name:
        return builder_cls(code=code)
    return builder_cls()


__all__ = [
    "get_builder",
    "CardinalBuilder",
    "ZXColorationBuilder",
    "FreeformBuilder",
    "CircuitBuildOptions",
]
