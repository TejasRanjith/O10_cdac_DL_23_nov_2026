import numpy as np

def single_layer_forward(X: np.ndarray, w: np.ndarray, b: float, activation_fn):
    """Compute pre-activation z = X @ w + b and post-activation a = g(z)."""
    z = X @ w + b
    a = activation_fn(z)
    return z, a

'''
1. what is forward pass for a single layer in a neural network?
   The forward pass for a single layer involves computing the pre-activation z = X @ w + b and then applying an activation function g to obtain the post-activation a = g(z).

'''

def step_1_forward_pass() -> None:
    print("=== Step 1: Forward Pass (z = X @ w + b) ===")
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=np.float64)
    w = np.array([1.5, -2.0], dtype=np.float64)
    b = 0.25

    # activation funtion in this is : identity (linear) function : lambda z: z
    z_linear, a_linear = single_layer_forward(X, w, b, lambda z: z)
    print("Inputs X:\n", X)
    print("Pre-activations z (identity):", np.round(z_linear, 4))
    print("Outputs a (identity):      ", np.round(a_linear, 4))
    print("_" * 60)

def step_2_linear_separability_and_xor() -> None:
    print("=== Step 2: Linear Separability vs. XOR ===")
    X = np.array([
        [0.0, 0.0], # marriage not allowed
        [0.0, 1.0], # marriage not allowed
        [1.0, 0.0], # marriage not allowed
        [1.0, 1.0], # marriage allowed
    ], dtype=np.float64)

    # And gate weights : z = 1.0 * x1 + 1.0 * x2 - 1.5 -> active (1) only when both x1 and x2 are 1
    w_and = np.array([1.0, 1.0], dtype=np.float64)
    b_and = -1.5
    _, pred_and = single_layer_forward(X, w_and, b_and, lambda z: (z >= 0.0).astype(int))
    target_and = np.array([0, 0, 0, 1], dtype=int)

    # XOR target : 1 when inputs differ (i.e., one is 1 and the other is 0)
    # target_xor = np.array([0,1,1,0], dtype=int)


    print("Predictions for AND gate:", pred_and)
    print("Target for AND gate:     ", target_and)
    # print("Target for XOR gate:     ", target_xor)


def step_3_unscaled_features() -> None:
    print("=== Step 3: Unscaled Features ===")
    X_normal = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ], dtype=np.float64)

    # Feature 2 is multiplied by 1000 (e.g., raw rupees vs small normalized feature)
    X_unscaled = np.array([
        [1.0, 2000.0],
        [2.0, 3000.0],
    ], dtype=np.float64)

    w = np.array([0.5, 0.5], dtype=np.float64)
    b = 0.0

    z_normal, _ = single_layer_forward(X_normal, w, b, lambda z: z)
    z_unscaled, _ = single_layer_forward(X_unscaled, w, b, lambda z: z)

    print("Pre-activation z (normal features):  ", z_normal)
    print("Pre-activation z (unscaled feature 2):", z_unscaled)
    print("Note: Large feature magnitudes dominate z even with equal weights.\n")


def step_4_zero_and_identical_weights() -> None:
    print("=== Step 4: Zero & Identical Weights ===")
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ], dtype=np.float64)

    w_zero = np.array([0.0, 0.0], dtype=np.float64)
    b_zero = 0.0
    z_zero, _ = single_layer_forward(X, w_zero, b_zero, lambda z: z)

    # Two units with identical weights and biases produce identical pre-activations
    w_unit_a = np.array([0.7, 0.7], dtype=np.float64)
    w_unit_b = np.array([0.7, 0.7], dtype=np.float64)
    z_a, _ = single_layer_forward(X, w_unit_a, 0.1, lambda z: z)
    z_b, _ = single_layer_forward(X, w_unit_b, 0.1, lambda z: z)

    print("z with zero weights & bias:     ", z_zero)
    print("Unit A == Unit B identical z?   ", np.allclose(z_a, z_b))
    print("Note: Identical parameters keep symmetric units locked in step (symmetry).\n")

def safe_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid without exp overflow for large |z|."""

    z = np.asarray(z, dtype=np.float64) # ensure numerical stability by converting to float64
    pos_mask = z >= 0 # Boolean mask for positive values of z
    out = np.empty_like(z, dtype=np.float64) # allocate output array with same shape and type as z
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    # z[~pos_mask] contains the negative values of z
    # np.exp : computes the exponential of the input array element-wise
    exp_z = np.exp(z[~pos_mask]) # compute exp(z) for negative values of z to avoid overflow in exp(-z)
    out[~pos_mask] = exp_z / (1.0 + exp_z)
    return out

def step_5_sigmoid_overflow() -> None:
    print("=== Step 5: Sigmoid Overflow vs. Stable Sigmoid ===")
    extreme_z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        naive_sig = 1.0 / (1.0 + np.exp(-extreme_z)) # sigma = 1 / (1 + exp(-z))

    stable_sig = safe_sigmoid(extreme_z)

    print("extreme_z:      ", extreme_z)
    print("Naive sigmoid:  ", naive_sig)
    print("Stable sigmoid: ", stable_sig)

if __name__ == "__main__":
    # step_1_forward_pass()
    # step_2_linear_separability_and_xor()
    # step_3_unscaled_features()
    # step_4_zero_and_identical_weights()
    step_5_sigmoid_overflow()
    