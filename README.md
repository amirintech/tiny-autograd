# Tiny Autograd

Tiny Autograd is a lightweight autograd engine built from scratch. It serves as an educational implementation of automatic differentiation, similar to PyTorch's autograd but in a minimalistic manner.

## Overview

This project implements a simple computational graph that enables automatic differentiation for scalar-valued functions. It provides core functionality for backpropagation. 

The repository includes an implementation of a basic neural network using the autograd engine to demonstrate its capabilities.

## Features

- Minimalistic automatic differentiation engine
- Supports forward and backward passes
- Enables basic optimization workflows
- Neural network implementation using Tiny Autograd

## Files

- `engine.py` - Core implementation of the autograd engine
- `nn.py` - Simple neural network implementation
- `train.py` - Training script demonstrating backpropagation
- `README.md` - Project documentation

## Usage

You can define simple mathematical operations and compute their gradients using backpropagation. Example usage:

```python
from engine import Value

# Define variables
a = Value(2.0)
b = Value(3.0)
c = a * b

d = c + Value(5.0)
d.backward()

print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
```

### Training a Simple Neural Network

A basic neural network is implemented in `nn.py`, and you can train it using `train.py`.

```bash
python train.py
```
