from .cardinal import CardinalBuilder
from .freeform import FreeformBuilder
from .xzcoloration import XZColorationBuilder

_BUILDERS = {
    CardinalBuilder.name: CardinalBuilder,
    XZColorationBuilder.name: XZColorationBuilder,
    FreeformBuilder.name: FreeformBuilder,
}


def get_builder(name):
    if name is None:
        name = "cardinal"
    builder_cls = _BUILDERS.get(name)
    if builder_cls is None:
        raise ValueError(f"Unknown circuit construction strategy: {name}")
    return builder_cls()


__all__ = ["get_builder", "CardinalBuilder", "XZColorationBuilder", "FreeformBuilder"]
