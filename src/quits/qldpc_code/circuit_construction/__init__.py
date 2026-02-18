from .cardinal import CardinalBuilder
from .cardinalNSmerge import CardinalNSMergeBuilder
from .circuit_build_options import CircuitBuildOptions
from .edge_coloration import edge_coloration
from .custom import CustomBuilder
from .zxcoloration import ZXColorationBuilder

_BUILDERS = {
    CardinalBuilder.name: CardinalBuilder,
    CardinalNSMergeBuilder.name: CardinalNSMergeBuilder,
    ZXColorationBuilder.name: ZXColorationBuilder,
    CustomBuilder.name: CustomBuilder,
}


def get_builder(name, code=None):
    if name is None:
        name = "cardinal"
    builder_cls = _BUILDERS.get(name)
    if builder_cls is None:
        raise ValueError(f"Unknown circuit construction strategy: {name}")
    if name in (CardinalBuilder.name, CardinalNSMergeBuilder.name, ZXColorationBuilder.name):
        return builder_cls(code=code)
    return builder_cls()


__all__ = [
    "get_builder",
    "CardinalBuilder",
    "CardinalNSMergeBuilder",
    "ZXColorationBuilder",
    "CustomBuilder",
    "CircuitBuildOptions",
    "edge_coloration",
]
