import random

from engine import Value


class Unit:
    def __init__(self, n_input, activation='linear'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x, ):
        z = sum([wi * xi for wi, xi in zip(self.w, x)], self.b)
        if self.activation == "linear":
            return z.linear()
        elif self.activation == "relu":
            return z.relu()
        elif self.activation == "leaky_relu":
            return z.leaky_relu()
        elif self.activation == "sigmoid":
            return z.sigmoid()
        elif self.activation == "tanh":
            return z.tanh()
        else:
            raise ValueError(f"Unsupported activation function '{self.activation}'")

    def params(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_input, n_output, activation='linear'):
        self.units = [Unit(n_input, activation) for _ in range(n_output)]
        self.activation = activation

    def __call__(self, x):
        return [u(x) for u in self.units]

    def params(self):
        return [p for u in self.units for p in u.params()]


class MLP:
    def __init__(self, n_input, layers, activations=None):
        sizes = [n_input] + layers
        self.activations = activations or ['linear'] * len(layers)
        self.layers = [Layer(sizes[i], sizes[i + 1], self.activations[i]) for i in range(len(layers))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for l in self.layers for p in l.params()]
