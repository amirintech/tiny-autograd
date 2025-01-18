import math
from math import isclose

import pytest

# import your Value class
from engine import Value


def test_value_creation():
    v = Value(2.0)
    assert v.data == 2.0
    assert v.grad == 0.0


@pytest.mark.parametrize("a,b,expected", [
    (2.0, 3.0, 5.0),
    (-1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.5, 2.5, 4.0)
])
def test_addition(a, b, expected):
    va = Value(a)
    vb = Value(b)
    vc = va + vb
    assert isclose(vc.data, expected, rel_tol=1e-9)

    vc.backward()
    # grad in va should accumulate partial derivative 1
    # grad in vb should accumulate partial derivative 1
    assert isclose(va.grad, 1.0, rel_tol=1e-9)
    assert isclose(vb.grad, 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("a,b,expected", [
    (2.0, 3.0, 6.0),
    (-1.0, 1.0, -1.0),
    (0.0, 5.0, 0.0),
    (1.5, 2.0, 3.0)
])
def test_multiplication(a, b, expected):
    va = Value(a)
    vb = Value(b)
    vc = va * vb
    assert isclose(vc.data, expected, rel_tol=1e-9)

    vc.backward()
    # grad in va should be b
    # grad in vb should be a
    assert isclose(va.grad, b, rel_tol=1e-9)
    assert isclose(vb.grad, a, rel_tol=1e-9)


def test_power():
    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0

    b.backward()
    # derivative of x^3 w.r.t x is 3x^2 => 3*2^2=12
    assert isclose(a.grad, 12.0, rel_tol=1e-9)


def test_activation_tanh():
    a = Value(0.0).tanh()
    assert isclose(a.data, 0.0, rel_tol=1e-9)

    a.backward()
    # derivative of tanh(x) at x=0 is 1 - tanh^2(0) = 1
    parent = a._children[0]
    assert isclose(parent.grad, 1.0, rel_tol=1e-9)


def test_activation_relu():
    a = Value(2.0).relu()
    b = Value(-2.0).relu()
    assert a.data == 2.0
    assert b.data == 0.0

    a.backward()
    # derivative at x=2 (positive) => 1
    parent_a = a._children[0]
    assert isclose(parent_a.grad, 1.0, rel_tol=1e-9)

    b.backward()
    # derivative at x=-2 (negative) => 0
    parent_b = b._children[0]
    assert isclose(parent_b.grad, 0.0, rel_tol=1e-9)


def test_activation_sigmoid():
    a = Value(0.0).sigmoid()
    assert isclose(a.data, 0.5, rel_tol=1e-9)

    a.backward()
    # derivative of sigmoid(0) = 0.5 * (1 - 0.5) = 0.25
    parent = a._children[0]
    assert isclose(parent.grad, 0.25, rel_tol=1e-9)


@pytest.mark.parametrize("x,alpha,expected", [
    (-2.0, 0.1, -0.2),
    (2.0, 0.1, 2.0),
    (0.0, 0.1, 0.0),
    (-1.0, 0.01, -0.01),
])
def test_activation_leaky_relu(x, alpha, expected):
    a = Value(x).leaky_relu(alpha=alpha)
    assert isclose(a.data, expected, rel_tol=1e-9)

    a.backward()
    parent = a._children[0]
    # derivative is alpha if x < 0 else 1
    if x > 0:
        assert isclose(parent.grad, 1.0, rel_tol=1e-9)
    else:
        assert isclose(parent.grad, alpha, rel_tol=1e-9)


def test_division():
    x = Value(6.0)
    y = Value(3.0)
    z = x / y
    assert isclose(z.data, 2.0, rel_tol=1e-9)

    z.backward()
    # z = x * (y^-1)
    # dz/dx = 1/y => 1/3
    # dz/dy = -x / y^2 => -6/9 = -2/3
    assert isclose(x.grad, 1 / 3, rel_tol=1e-9)
    assert isclose(y.grad, -2 / 3, rel_tol=1e-9)


def test_neg_sub():
    x = Value(3.0)
    y = Value(2.0)
    z = x - y  # 3 - 2 = 1
    assert isclose(z.data, 1.0, rel_tol=1e-9)

    z.backward()
    assert isclose(x.grad, 1.0, rel_tol=1e-9)
    assert isclose(y.grad, -1.0, rel_tol=1e-9)

    # test negative
    n = -x
    assert isclose(n.data, -3.0, rel_tol=1e-9)
    n.backward()
    # derivative of -x w.r.t x is -1
    assert isclose(x.grad, 0.0, rel_tol=1e-9), \
        "x.grad is 0 because 1 + (-1) = 0 (accumulated from both calls)"


def test_combined_operations_and_grad():
    # z = (x + y) * x
    x = Value(2.0)
    y = Value(3.0)
    z = (x + y) * x
    assert isclose(z.data, 10.0, rel_tol=1e-9)

    z.backward()
    # dz/dx = (x + y)*1 + x*1 => 2 + 3 + 2 = 7 if you expand carefully: x^2 + xy => 2*2 + 3
    # dz/dy = x
    assert isclose(x.grad, 7.0, rel_tol=1e-9)
    assert isclose(y.grad, 2.0, rel_tol=1e-9)


def test_chain_rule_gradients():
    x = Value(3.0)
    y = x ** 2
    z = y.sigmoid()

    z.backward()
    # manual check
    sigmoid_y = 1 / (1 + math.exp(-9.0))  # 9 = 3^2
    dy_dx = 2 * x.data  # 6
    dz_dy = sigmoid_y * (1 - sigmoid_y)
    dz_dx = dy_dx * dz_dy
    assert isclose(x.grad, dz_dx, rel_tol=1e-9)


def test_long_chain_numerical_check():
    # out = exp( relu( ((x+2)^2) / 5 ) )
    x = Value(2.0)
    out = (((x + 2.0) ** 2) / Value(5.0)).relu().exp()
    assert out.data > 0.0  # sanity check

    out.backward()
    autodiff_grad = x.grad

    # small numerical gradient check
    eps = 1e-5
    x_perturb = Value(x.data + eps)
    out_perturb = (((x_perturb + 2.0) ** 2) / Value(5.0)).relu().exp()
    numeric_grad = (out_perturb.data - out.data) / eps

    # allow a bit more tolerance due to chain of operations
    assert isclose(autodiff_grad, numeric_grad, rel_tol=1e-3), \
        f"mismatch: autodiff={autodiff_grad}, numerical={numeric_grad}"
