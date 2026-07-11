import numpy as np

# --- Activation Function ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)


#  --- Conducting forward propagation ---
def forward_propagation(x, thetas): # X is the training instance, thetas is the list of weights in the network
    a = np.concatenate(([1], x))

    activations = [a]

    for l, theta in enumerate(thetas):
        z = theta @ a # Matrix multiplication
        a = sigmoid(z)

        if l < len(thetas) - 1:
            a = np.concatenate(([1], a)) # If layer is not last layer, add bias and then append

        activations.append(a)

    return activations


# --- Computing the cost function ---

# Single instance
def compute_cost(y, output):
    return -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output))

# Regularize the data except bias
def compute_regularization(thetas, lam, n):
    reg = 0

    for theta in thetas:
        reg += np.sum(theta[:, 1:] ** 2)
    return (lam / (2*n)) * reg

# Total average cost of instances + regularized weights
def compute_total_cost(x_list, y_list, thetas, lam):
    n = len(x_list)
    total = 0

    for x, y in zip(x_list, y_list):
        activations = forward_propagation(x, thetas)
        output = activations[-1] # the last layer of the network is the output
        total += compute_cost(y, output)

    total = total / n
    total += compute_regularization(thetas, lam, n)

    return total



# --- Conducting Backpropagation ---

def backpropagation(x_list, y_list, thetas, lam):
    n = len(x_list)
    big_delta = [np.zeros_like(theta) for theta in thetas]

    for x, y in zip(x_list, y_list):
        activations = forward_propagation(x, thetas)
        deltas = [None] * len(thetas)

        deltas[-1] = activations[-1] - y # Output layer

        for l in range(len(thetas) - 2, -1, -1):
            a_hidden = activations[l + 1][1:]  # strip bias
            
            delta_no_bias = (thetas[l + 1].T @ deltas[l + 1])[1:]
            
            deltas[l] = delta_no_bias * sigmoid_derivative(a_hidden)


        for l in range(len(thetas)):
            big_delta[l] += np.outer(deltas[l], activations[l])


    gradients = []

    for l, theta in enumerate(thetas):
        D = big_delta[l] / n

        reg = (lam / n) * theta
        reg[:, 0] = 0  # zero out bias column so it's not regularized

        gradients.append(D + reg)

    return gradients


# --- Weight Initialization ---

def initialize_thetas(layer_sizes):
    thetas = []

    for i in range(len(layer_sizes) - 1):
        rows = layer_sizes[i + 1] # neurons in next layer
        cols = layer_sizes[i] + 1 # neurons in current layer + bias
        
        theta = np.random.uniform(-1, 1, (rows, cols))
        thetas.append(theta)
    return thetas


# --- Training the Network ---

def train(x_list, y_list, layer_sizes, lam, alpha, epsilon, max_iter=10000):
    thetas = initialize_thetas(layer_sizes)
    cost_history = []

    J = compute_total_cost(x_list, y_list, thetas, lam)
    cost_history.append(J)

    for iteration in range(max_iter):
        gradients = backpropagation(x_list, y_list, thetas, lam)

        for l in range(len(thetas)):
            thetas[l] = thetas[l] - alpha * gradients[l]

        J_new = compute_total_cost(x_list, y_list, thetas, lam)
        cost_history.append(J_new)

        if abs(J - J_new) < epsilon:
            print(f"Converged at iteration {iteration + 1}")
            break

        J = J_new

    else:
        print(f"Reached max iterations ({max_iter})")

    return thetas, cost_history


# --- Verification Function ---

def verify_backprop(x_list, y_list, thetas, lam):
    n = len(x_list)

    # --- Print network structure ---
    print(f"Regularization parameter lambda={lam:.3f}")
    structure = [x_list[0].shape[0]] + [t.shape[0] for t in thetas]
    print(f"Network structure: {structure}")
    print()

    for l, theta in enumerate(thetas):
        print(f"Theta{l+1}:")

        for row in theta:
            print("  " + "  ".join(f"{w:.5f}" for w in row))
        print()

    # --- Forward pass + cost per instance ---
    print("Computing the error/cost, J, of the network")

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        print(f"\n  Processing training instance {i+1}")
        print(f"  Forward propagating the input {x}")

        a = np.concatenate(([1.0], x))
        print(f"    a1: {a}")

        activations = [a]

        for l, theta in enumerate(thetas):
            z = theta @ a
            a = sigmoid(z)
            print(f"\n    z{l+2}: {z}")

            if l < len(thetas) - 1:
                a = np.concatenate(([1.0], a))

            print(f"    a{l+2}: {a}")
            activations.append(a)

        output = activations[-1]

        print(f"\n    f(x): {output}")
        print(f"  Predicted output for instance {i+1}: {output}")
        print(f"  Expected output for instance {i+1}: {y}")

        J = compute_cost(y, output)
        print(f"  Cost, J, associated with instance {i+1}: {J:.3f}")

    total_J = compute_total_cost(x_list, y_list, thetas, lam)
    print(f"\nFinal (regularized) cost, J: {total_J:.5f}")

    # --- Backpropagation per instance ---
    print("\nRunning backpropagation")
    big_delta = [np.zeros_like(theta) for theta in thetas]

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        print(f"\n  Computing gradients based on training instance {i+1}")

        # Forward pass to get activations
        a = np.concatenate(([1.0], x))
        activations = [a]

        for l, theta in enumerate(thetas):
            z = theta @ a
            a = sigmoid(z)
            if l < len(thetas) - 1:
                a = np.concatenate(([1.0], a))
            activations.append(a)

        # Deltas
        deltas = [None] * len(thetas)
        deltas[-1] = activations[-1] - y
        print(f"    delta{len(thetas)+1}: {deltas[-1]}")

        for l in range(len(thetas) - 2, -1, -1):
            a_hidden = activations[l + 1][1:]
            delta_no_bias = (thetas[l + 1].T @ deltas[l + 1])[1:]
            deltas[l] = delta_no_bias * sigmoid_derivative(a_hidden)
            print(f"    delta{l+2}: {deltas[l]}")

        # Gradients per instance
        print()
        for l in range(len(thetas) - 1, -1, -1):
            grad = np.outer(deltas[l], activations[l])
            big_delta[l] += grad
            print(f"    Gradients of Theta{l+1} based on instance {i+1}:")

            for row in grad:
                print("      " + "  ".join(f"{w:.5f}" for w in row))

            print()

    # --- Final regularized gradients ---
    print("  Final regularized gradients:")

    for l, theta in enumerate(thetas):
        D = big_delta[l] / n
        reg = (lam / n) * theta
        reg[:, 0] = 0
        grad_final = D + reg

        print(f"    Theta{l+1}:")

        for row in grad_final:
            print("      " + "  ".join(f"{w:.5f}" for w in row))

        print()


if __name__ == "__main__":
    # ── Example 1 ──
    print("=" * 50)
    print("EXAMPLE 1")
    print("=" * 50)
    theta1 = np.array([[0.4, 0.1],
                       [0.3, 0.2]])
    theta2 = np.array([[0.7, 0.5, 0.6]])
    thetas = [theta1, theta2]
    x_list = [np.array([0.13]), np.array([0.42])]
    y_list = [np.array([0.90]), np.array([0.23])]
    verify_backprop(x_list, y_list, thetas, lam=0)

    # ── Example 2 ──
    print("=" * 50)
    print("EXAMPLE 2")
    print("=" * 50)
    theta1 = np.array([[0.42, 0.15, 0.40],
                       [0.72, 0.10, 0.54],
                       [0.01, 0.19, 0.42],
                       [0.30, 0.35, 0.68]])
    theta2 = np.array([[0.21, 0.67, 0.14, 0.96, 0.87],
                       [0.87, 0.42, 0.20, 0.32, 0.89],
                       [0.03, 0.56, 0.80, 0.69, 0.09]])
    theta3 = np.array([[0.04, 0.87, 0.42, 0.53],
                       [0.17, 0.10, 0.95, 0.69]])
    thetas = [theta1, theta2, theta3]
    x_list = [np.array([0.32, 0.68]), np.array([0.83, 0.02])]
    y_list = [np.array([0.75, 0.98]), np.array([0.75, 0.28])]
    verify_backprop(x_list, y_list, thetas, lam=0.25)