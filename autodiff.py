import numpy as np
from scipy.special import softmax as scipy_softmax 

# ==========================================
# 1. Core Classes (Op and Var)
# ==========================================

class Op:
    def __init__(self, name, num_inputs, forward_fn, backward_fn):
        self.name = name
        self.num_inputs = num_inputs
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def __call__(self, *inputs):
        # 1. check that all inputs are Var instances
        # (Auto-wrap constants if they aren't Vars to make it user-friendly)
        inputs = [i if isinstance(i, Var) else Var(i) for i in inputs]

        # 2. get the raw values from the parent Var objects
        input_values = [v.value for v in inputs]

        # 3. perform the forward computation
        output_value = self.forward_fn(*input_values)

        # 4. return a new Var instance, linking it to its op and parents
        return Var(value=output_value, op=self, parents=list(inputs))

class Var:
    """
    Represents a variable in the computation graph.
    """
    def __init__(self, value, op=None, parents=None):
        self.value = np.array(value)
        self.op = op          # the Op that created this Var
        self.parents = parents if parents is not None else [] # list of parent Var instances
        self.grad = None      # stores gradient

    def __repr__(self):
        return f"Var(value={self.value}, op={self.op.name if self.op else 'Constant'})"

    # Operator overloading so you can do x + y instead of add(x, y)
    def __add__(self, other): return add(self, other)
    def __sub__(self, other): return sub(self, other)
    def __mul__(self, other): return mul(self, other)
    def __matmul__(self, other): return matmul(self, other)


# ==========================================
# 2. Helper Functions
# ==========================================

def safe_sum_for_broadcast(grad_output, input_shape):
    """
    Sums the output gradient 'grad_output' so its shape
    matches the 'input_shape' it's being backpropagated to.
    """
    # 1. if shapes match, do nothing
    if grad_output.shape == input_shape:
        return grad_output

    # 2. if input was a scalar (0-dim), sum grad completely
    if input_shape == ():
        return np.sum(grad_output)

    # 3. handle non-scalar input
    # sum over extra dims
    ndim_diff = grad_output.ndim - len(input_shape)
    if ndim_diff > 0:
        grad_output = np.sum(grad_output, axis=tuple(range(ndim_diff)))

    # sum over broadcasted dims (where input_dim=1)
    keep_dims_axes = tuple([i for i, dim in enumerate(input_shape) if dim == 1])
    if keep_dims_axes:
        grad_output = np.sum(grad_output, axis=keep_dims_axes, keepdims=True)

    # 4. reshape to match input shape
    return np.reshape(grad_output, input_shape)


# ==========================================
# 3. Backward Implementations
# ==========================================

def add_backward(u, x, y):
    dx = safe_sum_for_broadcast(u, x.shape)
    dy = safe_sum_for_broadcast(u, y.shape)
    return [dx, dy]

def sub_backward(u, x, y):
    dx = safe_sum_for_broadcast(u, x.shape)
    dy = safe_sum_for_broadcast(-u, y.shape)
    return [dx, dy]

def mul_backward(u, x, y):
    dx = safe_sum_for_broadcast(u * y, x.shape)
    dy = safe_sum_for_broadcast(u * x, y.shape)
    return [dx, dy]

def matmul_backward(u, X, Y):
    if u.ndim == 0:
        raise ValueError(f"MATMUL_BACKWARD: upstream grad 'u' is 0D. Cannot perform '@'.")

    # handle 1D vector cases
    grad_X_u = u
    grad_X_Y_T = Y.T

    if u.ndim == 1:
        grad_X_u = u.reshape(-1, 1)
    if Y.ndim == 1:
        grad_X_Y_T = Y.reshape(1, -1)

    grad_X = grad_X_u @ grad_X_Y_T
    grad_Y = X.T @ u

    return [grad_X, grad_Y]

def solve_backward(u, A, x):
    if u.ndim < 1:
        raise ValueError(f"SOLVE_BACKWARD: upstream grad 'u' is 0-dimensional.")

    b = np.linalg.solve(A, x)
    v = np.linalg.solve(A.T, u)

    v_for_matmul = v
    if v.ndim == 1:
        v_for_matmul = v.reshape(-1, 1)

    b_T_for_matmul = b.T
    if b.ndim == 1:
        b_T_for_matmul = b.reshape(1, -1)

    dA = -v_for_matmul @ b_T_for_matmul
    dx = v.reshape(x.shape)

    return [dA, dx]

def logdet_backward(u, A):
    return [u * np.linalg.inv(A.T)]

def sum_backward(u, x):
    return [np.full_like(x, u)]

def exp_backward(u, x):
    return [u * np.exp(x)]

def log_backward(u, x):
    return [u / x]

def logsumexp_backward(u, x):
    return [u * scipy_softmax(x)]


# ==========================================
# 4. Operator Instantiation
# ==========================================

add = Op("add", 2, lambda x, y: x + y, add_backward)
sub = Op("sub", 2, lambda x, y: x - y, sub_backward)
mul = Op("mul", 2, lambda x, y: x * y, mul_backward)
matmul = Op("matmul", 2, lambda x, y: x @ y, matmul_backward)
solve = Op("solve", 2, np.linalg.solve, solve_backward)
logdet = Op("logdet", 1, lambda A: np.linalg.slogdet(A)[1], logdet_backward)
exp = Op("exp", 1, np.exp, exp_backward)
log = Op("log", 1, np.log, log_backward)
_sum = Op("sum", 1, np.sum, sum_backward) 


# ==========================================
# 5. Backpropagation Engine
# ==========================================

def backpropagation(root_node, input_nodes):
    """
    Performs Reverse-Mode AD.
    """
    # 1. topological Sort
    topo_order = []
    visited = set()
    
    def build_topo(node):
        if node not in visited:
            visited.add(node)
            for parent in node.parents:
                build_topo(parent)
            topo_order.append(node)
    
    build_topo(root_node)
    
    # 2. initialize gradients
    root_node.grad = np.ones_like(root_node.value, dtype=float)
    
    # 3. backward pass
    for node in reversed(topo_order):
        if node.op:
            # get upstream gradient
            upstream_grad = node.grad
            
            # get parent values for backward fn
            parent_values = [p.value for p in node.parents]
            
            # compute gradients for parents
            parent_grads = node.op.backward_fn(upstream_grad, *parent_values)
            
            # accumulate
            for parent, g in zip(node.parents, parent_grads):
                if parent.grad is None:
                    parent.grad = np.zeros_like(parent.value, dtype=float)
                parent.grad += g
                
    return [n.grad for n in input_nodes]


def grad(computation_function):
    """
    Wrapper that returns a function computing gradients.
    """
    def grad_f(*args):
        # 1. convert inputs to Var
        input_vars = [Var(arg) for arg in args]

        # 2. run forward pass
        output_node = computation_function(*input_vars)

        # 3. run backward pass
        gradients = backpropagation(output_node, input_vars)

        return gradients

    return grad_f