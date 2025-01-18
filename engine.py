import math


class Value:
    def __init__(self, data, _children=(), _operation='', label=''):
        self.data = data
        self.grad = 0.0  # gradient
        self.label = label  # for debugging
        self._children = _children
        self._backward = lambda: None  # for calculating gradient
        self._operation = _operation  # for debugging

    def backward(self):
        nodes = []
        visited = set()

        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build_graph(c)
                nodes.append(v)

        build_graph(self)
        self.grad = 1.0
        for n in reversed(nodes):
            n._backward()

    # activation functions
    def linear(self):
        out = Value(self.data, _children=(self,), _operation='linear')

        def _backward():
            self.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def tanh(self):
        data = math.tanh(self.data)
        out = Value(data, _children=(self,), _operation='tanh')

        def _backward():
            self.grad += out.grad * (1 - data ** 2)

        out._backward = _backward
        return out

    def leaky_relu(self, alpha=0.01):
        data = self.data if self.data > 0 else self.data * alpha
        out = Value(data, _children=(self,), _operation='leaky_relu')

        def _backward():
            self.grad += out.grad * (1 if self.data > 0 else alpha)

        out._backward = _backward
        return out

    def relu(self):
        data = max(0, self.data)
        out = Value(data, _children=(self,), _operation='relu')

        def _backward():
            self.grad += out.grad * (data > 0)

        out._backward = _backward
        return out

    def sigmoid(self):
        data = 1 / (1 + math.exp(-self.data))
        out = Value(data, _children=(self,), _operation='sigmoid')

        def _backward():
            self.grad += out.grad * data * (1 - data)

        out._backward = _backward
        return out

    def exp(self):
        data = math.exp(self.data)
        out = Value(data, _children=(self,), _operation='exp')

        def _backward():
            self.grad += out.grad * data

        out._backward = _backward
        return out

    # basic math operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self.data + other.data
        out = Value(data, _children=(self, other), _operation='add')

        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self.data * other.data
        out = Value(data, _children=(self, other), _operation='mul')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), f"unsupported type '{type(power)}'"
        data = self.data ** power
        out = Value(data, _children=(self,), _operation=f'pow {power}')

        def _backward():
            self.grad += out.grad * power * self.data ** (power - 1)

        out._backward = _backward
        return out

    def __radd__(self, other):
        assert isinstance(other, (int, float)), f"unsupported type '{type(other)}'"
        return self.__add__(other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float)), f"unsupported type '{type(other)}'"
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other):
        assert isinstance(other, (int, float, Value)), f"unsupported type '{type(other)}'"
        other = Value(other) if isinstance(other, (int, float)) else other
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        assert isinstance(other, (int, float, Value)), f"unsupported type '{type(other)}'"
        other = Value(other) if isinstance(other, (int, float)) else other
        return self.__mul__(other.__pow__(-1))

    # internal
    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"
