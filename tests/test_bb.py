import numpy as np

from quits.qldpc_code import BbCode


def test_bb_code_css_logicals():
    code = BbCode(
        l=15,
        m=3,
        A_x_pows=[9],
        A_y_pows=[1, 2],
        B_x_pows=[2, 7],
        B_y_pows=[0],
    )

    assert code.hx.shape == (45, 90)
    assert code.hz.shape == (45, 90)
    assert np.all((code.hx == 0) | (code.hx == 1))
    assert np.all((code.hz == 0) | (code.hz == 1))

    assert not ((code.hx @ code.hz.T) & 1).any()

    report = code.verify_css_logicals()
    assert report["all_tests_passed"]

    return {
        "hx_shape": code.hx.shape,
        "hz_shape": code.hz.shape,
        "all_tests_passed": report["all_tests_passed"],
        "verify_css_logicals": report,
    }


if __name__ == "__main__":
    result = test_bb_code_css_logicals()
    print("verify_css_logicals", result["verify_css_logicals"])
    print("test_bb_code_css_logicals", result)
