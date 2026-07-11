# Neural Network from Scratch

A fully vectorized implementation of backpropagation for training neural networks, built from scratch in Python (NumPy only — no ML libraries like TensorFlow, PyTorch, or scikit-learn's models).

## Overview

This project implements:
- **Backpropagation** with support for configurable architectures (variable number of hidden layers and neurons per layer)
- **L2 regularization** to mitigate overfitting, with bias weights correctly excluded from regularization
- **Stratified k-fold cross-validation** (k=5) for model evaluation
- **Gradient correctness verification** against provided reference outputs

The network was evaluated on two datasets:
- **WDBC** (Wisconsin Breast Cancer Diagnostic) — 569 instances, 30 numerical features, binary classification (malignant/benign)
- **Loan Eligibility Prediction** — 480 instances, 11 features (7 categorical, one-hot encoded), binary classification (loan approved/denied)

## Repository Structure

```
├── src/
│   ├── neural_network.py     # Core backpropagation implementation + verification entry point
│   ├── data.py                # Data loading, preprocessing, one-hot encoding
│   ├── experiments.py         # Cross-validation and architecture sweep logic
│   └── run_experiments.py     # Script to reproduce reported results
├── datasets/                  # WDBC, Loan, Raisin, Titanic datasets
├── verification/               # Reference backprop outputs used for correctness checks
├── figures/                    # Learning curve plots
└── HW4_Report.pdf              # Full write-up with methodology, results, and analysis
```

## Verifying Correctness

The implementation was validated by reproducing the exact intermediate quantities (activations, deltas, gradients) from two reference neural networks. To run this verification:

```bash
python neural_network.py
```

This calls `verify_backprop` against `backprop_example1.txt` and `backprop_example2.txt`, printing forward-propagation activations, cost J, delta values, per-instance gradients, and final regularized gradients — all matched against the provided reference values.

## Results

Each dataset was evaluated across 6 architectures using stratified 5-fold CV, with λ=0, α=0.5, and stopping criterion ε=10⁻⁴.

**WDBC**

| Architecture | Accuracy | F1-Score |
|---|---|---|
| [30, 4, 1] | 0.9701 | 0.9764 |
| [30, 8, 1] | 0.9631 | 0.9711 |
| [30, 16, 1] | 0.9666 | 0.9735 |
| [30, 4, 4, 1] | 0.8941 | 0.9291 |
| [30, 8, 4, 1] | 0.8350 | 0.8958 |
| [30, 8, 8, 1] | 0.9649 | 0.9722 |

**Loan**

| Architecture | Accuracy | F1-Score |
|---|---|---|
| [21, 4, 1] | 0.8066 | 0.8751 |
| [21, 8, 1] | 0.8083 | 0.8761 |
| [21, 16, 1] | 0.8062 | 0.8746 |
| [21, 4, 4, 1] | 0.7416 | 0.8432 |
| [21, 8, 4, 1] | 0.7376 | 0.8400 |
| [21, 8, 8, 1] | 0.7892 | 0.8674 |

**Key finding:** deeper networks consistently underperformed shallow ones on both datasets, often converging to bad local minima. A single hidden layer with a small number of neurons ([30, 4, 1] for WDBC, [21, 8, 1] for Loan) gave the best results — added depth hurt generalization rather than helping, likely due to the relatively small dataset sizes.

## Learning Curves

Learning curves (cost J on a held-out test set vs. number of training instances) were generated for the best architecture on each dataset:
- **WDBC** [30, 4, 1]: J dropped from 0.51 (n=30) to 0.13 (n=450), improving consistently with more data.
- **Loan** [21, 4, 1]: J dropped from 1.09 (n=30) to 0.44 (n=360), flattening after n≈150.

See `figures/learning_curve_wdbc.png` and `figures/learning_curve_loan.png`.

## Notes

- Full-batch gradient descent was used throughout (mini-batch was not implemented).
- Full methodology, architecture selection reasoning, and extended discussion are in `HW4_Report.pdf`.