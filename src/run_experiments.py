import matplotlib.pyplot as plt
from data import load_wdbc, load_loan
from experiments import run_experiments, learning_curve

# ── Load datasets ──
x_w, y_w = load_wdbc("datasets/wdbc.csv")
x_l, y_l = load_loan("datasets/loan.csv")

architectures = [[4], [8], [16], [4, 4], [8, 4], [8, 8]]
lambdas = [0]

# ── WDBC experiments ──
print("=" * 60)
print("WDBC DATASET")
print("=" * 60)
results_w = run_experiments(x_w, y_w, input_size=30, architectures=architectures, lambdas=lambdas, alpha=0.5, epsilon=1e-4, k=5)

# ── Loan experiments ──
print("\n" + "=" * 60)
print("LOAN DATASET")
print("=" * 60)
results_l = run_experiments(x_l, y_l, input_size=21, architectures=architectures, lambdas=lambdas, alpha=0.5, epsilon=1e-4, k=5)

# ── Learning curves ──
print("\nGenerating learning curve for WDBC...")
sizes_w, costs_w = learning_curve(x_w, y_w, layer_sizes=[30, 4, 1], lam=0, alpha=0.5, epsilon=1e-4, step=30)

plt.figure()
plt.plot(sizes_w, costs_w, marker='o')

plt.xlabel("Number of training instances")
plt.ylabel("Cost J (test set)")
plt.title("Learning Curve - WDBC [30, 4, 1]")
plt.grid(True)

plt.savefig("learning_curve_wdbc.png")


print("\nGenerating learning curve for Loan...")
sizes_l, costs_l = learning_curve(x_l, y_l, layer_sizes=[21, 4, 1], lam=0, alpha=0.5, epsilon=1e-4, step=30)

plt.figure()
plt.plot(sizes_l, costs_l, marker='o')

plt.xlabel("Number of training instances")
plt.ylabel("Cost J (test set)")
plt.title("Learning Curve - Loan [21, 4, 1]")
plt.grid(True)

plt.savefig("learning_curve_loan.png")