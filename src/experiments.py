# experiments.py
import numpy as np
from neural_network import train, forward_propagation

def stratified_kfold_split(y_list, k=5):
    class_indices = {}

    for i, y in enumerate(y_list):
        label = int(y[0])

        if label not in class_indices:
            class_indices[label] = []
            
        class_indices[label].append(i)

    # Shuffle indices within each class for randomness
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Split each class into k chunks
    class_chunks = {}
    for label, indices in class_indices.items():
        class_chunks[label] = [indices[i::k] for i in range(k)]

    # Build k folds by combining one chunk from each class
    folds = []

    for fold_idx in range(k):
        fold = []

        for label in class_chunks:
            fold.extend(class_chunks[label][fold_idx])

        folds.append(fold)

    return folds


def evaluate(x_list, y_list, thetas, threshold=0.5):
    tp = fp = tn = fn = 0

    for x, y in zip(x_list, y_list):
        activations = forward_propagation(x, thetas)
        output = activations[-1][0]

        predicted = 1 if output >= threshold else 0
        actual = int(y[0])

        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return accuracy, f1


def cross_validate(x_list, y_list, layer_sizes, lam, alpha, epsilon, k=5):
    folds = stratified_kfold_split(y_list, k)
    all_indices = list(range(len(x_list)))

    accuracies = []
    f1_scores = []

    for fold_idx in range(k):
        test_indices = folds[fold_idx]
        train_indices = [i for i in all_indices if i not in set(test_indices)]

        x_train = [x_list[i] for i in train_indices]
        y_train = [y_list[i] for i in train_indices]
        x_test  = [x_list[i] for i in test_indices]
        y_test  = [y_list[i] for i in test_indices]

        # Train network on training fold
        thetas, _ = train(x_train, y_train, layer_sizes, lam, alpha, epsilon)

        # Evaluate on test fold
        acc, f1 = evaluate(x_test, y_test, thetas)
        accuracies.append(acc)
        f1_scores.append(f1)

        print(f"  Fold {fold_idx + 1}: Accuracy={acc:.4f}, F1={f1:.4f}")

    mean_accuracy = np.mean(accuracies)
    mean_f1 = np.mean(f1_scores)

    return mean_accuracy, mean_f1


# Simulate experiments
def run_experiments(x_list, y_list, input_size, architectures, lambdas, alpha, epsilon, k=5):
    results = []

    for hidden_layers in architectures:
        for lam in lambdas:
            layer_sizes = [input_size] + hidden_layers + [1]
            print(f"\nArchitecture: {layer_sizes}, Lambda: {lam}")

            acc, f1 = cross_validate(x_list, y_list, layer_sizes, lam, alpha, epsilon, k)

            print(f"  --> Mean Accuracy: {acc:.4f}, Mean F1: {f1:.4f}")

            results.append({
                'architecture': layer_sizes,
                'lambda': lam,
                'accuracy': acc,
                'f1': f1
            })

    return results


def learning_curve(x_list, y_list, layer_sizes, lam, alpha, epsilon, step=5):
    from neural_network import compute_total_cost

    # Use 80% for train pool, 20% as fixed test set
    split = int(0.8 * len(x_list))
    x_train_pool = x_list[:split]
    y_train_pool = y_list[:split]
    x_test = x_list[split:]
    y_test = y_list[split:]

    sizes = []
    costs = []

    for n in range(step, len(x_train_pool) + 1, step):
        x_train = x_train_pool[:n]
        y_train = y_train_pool[:n]

        thetas, _ = train(x_train, y_train, layer_sizes, lam, alpha, epsilon)

        J = compute_total_cost(x_test, y_test, thetas, lam)
        sizes.append(n)
        costs.append(J)

        print(f"  n={n}, J={J:.4f}")

    return sizes, costs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data import load_wdbc

    x_w, y_w = load_wdbc("datasets/wdbc.csv")

    print("\nGenerating learning curve for WDBC...")
    sizes, costs = learning_curve(x_w, y_w, layer_sizes=[30, 4, 1], lam=0, alpha=0.5, epsilon=1e-4, step=30)

    plt.figure()
    plt.plot(sizes, costs, marker='o')

    plt.xlabel("Number of training instances")
    plt.ylabel("Cost J (test set)")
    plt.title("Learning Curve - WDBC [30, 4, 1]")
    plt.grid(True)

    plt.savefig("figures/learning_curve_wdbc.png")


    from data import load_loan

    x_l, y_l = load_loan("datasets/loan.csv")

    print("\nGenerating learning curve for Loan...")
    sizes_l, costs_l = learning_curve(x_l, y_l, layer_sizes=[21, 4, 1], lam=0, alpha=0.5, epsilon=1e-4, step=30)

    plt.figure()
    plt.plot(sizes_l, costs_l, marker='o')

    plt.xlabel("Number of training instances")
    plt.ylabel("Cost J (test set)")
    plt.title("Learning Curve - Loan [21, 4, 1]")
    plt.grid(True)

    plt.savefig("figures/learning_curve_loan.png")