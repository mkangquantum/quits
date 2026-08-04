"""GF(2) linear algebra, CSS utilities, group algebra, and lifted products.

Two layers:

- GF(2) matrix routines and CSS logical-operator utilities (:mod:`.linalg`,
  :mod:`.css`), adapted in part from the QuantumGizmos ldpc package
  (https://github.com/quantumgizmos/ldpc).
- Finite groups / F2[G] group algebra and the lifted-product construction
  (:mod:`.group`, :mod:`.lifted_product`), following arXiv:2607.28795.

The full public API is re-exported flat, so ``from quits.gf2_util import X``
works for every name below.
"""

from .linalg import (
    _as_gf2,
    _gf2_inv_square,
    gf2_coset_reps_rowspace,
    gf2_nullspace_basis,
    gf2_rank,
    gf2_row_basis,
    gf2_row_span,
    gf2_rref,
    gf2_solve,
    gf2_solve_square,
    in_rowspace,
)
from .css import (
    check_css,
    check_weights,
    compute_code_distance,
    compute_k,
    compute_lz,
    compute_lz_and_lx,
    qubit_weights,
    verify_css_logicals,
)
from .group import (
    MultTableGroup,
    alternating,
    canonicalize,
    cyclic,
    dagger,
    dihedral,
    direct_product,
    element_order,
    generated_subgroup,
    is_self_dagger,
    left_rep,
    matrix_dagger,
    right_rep,
    ring_add,
    ring_mul,
    symmetric,
)
from .lifted_product import (
    build_Hx_from_lifted_product,
    build_Hz_from_lifted_product,
    get_circulant_mat,
    lift,
    lift_enc,
)

__all__ = [
    # linalg
    "gf2_coset_reps_rowspace",
    "gf2_nullspace_basis",
    "gf2_rank",
    "gf2_row_basis",
    "gf2_row_span",
    "gf2_rref",
    "gf2_solve",
    "gf2_solve_square",
    "in_rowspace",
    # css
    "check_css",
    "check_weights",
    "compute_code_distance",
    "compute_k",
    "compute_lz",
    "compute_lz_and_lx",
    "qubit_weights",
    "verify_css_logicals",
    # group
    "MultTableGroup",
    "alternating",
    "canonicalize",
    "cyclic",
    "dagger",
    "dihedral",
    "direct_product",
    "element_order",
    "generated_subgroup",
    "is_self_dagger",
    "left_rep",
    "matrix_dagger",
    "right_rep",
    "ring_add",
    "ring_mul",
    "symmetric",
    # lifted_product
    "build_Hx_from_lifted_product",
    "build_Hz_from_lifted_product",
    "get_circulant_mat",
    "lift",
    "lift_enc",
]
