# Mini-AutoDiff: A Lightweight Reverse-Mode AD Engine

A lightweight **Automatic Differentiation (AD)** library built from scratch in Python and NumPy. This project implements **Reverse-Mode Differentiation (Backpropagation)** using a dynamic computational graph, enabling gradient computation for complex linear algebra operations without relying on heavy frameworks like PyTorch or TensorFlow.

## 🚀 Features

* **Reverse-Mode AD:** Efficiently computes gradients for scalar and tensor operations via the Chain Rule.
* **Dynamic Computational Graph:** Constructs graphs on-the-fly (similar to PyTorch's `autograd`).
* **Linear Algebra Support:** Includes custom backward definitions for complex matrix operations:
    * Matrix Multiplication (`@`)
    * Linear Solve (`np.linalg.solve`)
    * Log Determinant (`np.linalg.slogdet`)
* **Broadcasting Support:** Handles gradient shape matching automatically for broadcasted operations.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/mini-autodiff.git](https://github.com/YOUR-USERNAME/mini-autodiff.git)
    cd mini-autodiff
    ```

2.  **Install dependencies**
    This project requires `numpy` and `scipy`.
    ```bash
    pip install numpy scipy
    ```

## ⚡ Quick Start

You can use the `autodiff` library to define scalar or matrix functions and compute their gradients automatically.

```python
import numpy as np
import autodiff as ad

# 1. Define inputs
x = 2.0
y = 3.0

# 2. Define a computation function
def my_func(x, y):
    # z = x * y + x
    return x * y + x 

# 3. Compute Gradients
# ad.grad() returns a function that computes the gradient w.r.t inputs
grad_fn = ad.grad(my_func)
grads = grad_fn(x, y)

print(f"Gradient dx: {grads[0]}")  # Output: 4.0 (y + 1)
print(f"Gradient dy: {grads[1]}")  # Output: 2.0 (x)
```

## 🧠 Advanced Usage: Multivariate Gaussian

The engine is robust enough to handle complex statistical models. The included `demo.ipynb` showcases computing the gradient of the **Negative Log-Likelihood** for a Multivariate Gaussian distribution with respect to the Covariance Matrix $\Sigma$.

$$\mathcal{L}(\Sigma) = \frac{N}{2} \log |\Sigma| + \frac{1}{2} \sum_{i=1}^{N} (x_i^\top \Sigma^{-1} x_i)$$

The engine correctly handles the gradients for `logdet` and `solve`:
* $\frac{\partial \log|\Sigma|}{\partial \Sigma} = (\Sigma^{-1})^\top$
* Backpropagation through linear systems $Ax=b$.

Check `demo.ipynb` for the full implementation and numerical verification against finite differences.

## 📂 Project Structure

* `autodiff.py`: The core library containing the `Var` class, `Op` definitions, and the topological sort backpropagation engine.
* `demo.ipynb`: A Jupyter Notebook demonstrating usage and validating correctness against numerical gradients.

## 📚 Technical Details

This implementation uses **Operator Overloading** to build a Directed Acyclic Graph (DAG) as operations are performed. When `grad()` is called:
1.  **Topological Sort:** The graph is traversed to ensure we process nodes in dependency order.
2.  **Vector-Jacobian Product (VJP):** Gradients are propagated backward from the output to the inputs using defined VJPs for each operator.

