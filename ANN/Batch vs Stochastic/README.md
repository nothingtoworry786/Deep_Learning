# Gradient Descent: Batch vs Stochastic

## Overview

Gradient Descent is an optimization algorithm used to minimize a loss function by updating model parameters.

There are two main types:

* Batch Gradient Descent
* Stochastic Gradient Descent (SGD)

---

## 🧠 Batch Gradient Descent

Batch Gradient Descent computes the gradient using the entire dataset before updating the parameters.

### Update Rule

θ(t+1)=θ(t)−η∇θ​J(θ(t))

### Characteristics

* Uses the full dataset for each update
* One update per epoch
* Smooth and stable convergence

### Advantages

* Stable updates
* Accurate convergence

### Disadvantages

* Slow for large datasets
* Requires high memory

---

## ⚡ Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent updates parameters using one training example at a time.

### Update Rule

θ(t+1)=θ(t)−η∇θ​J(θ(t);xi​,yi​)

### Characteristics

* Updates for every data point
* Many updates per epoch
* Noisy and fluctuating path

### Advantages

* Faster updates
* Works well for large datasets
* Can escape local minima

### Disadvantages

* Less stable
* May not converge exactly
* Oscillates near minimum

---

## 🔍 Comparison

| Feature              | Batch GD       | SGD         |
| -------------------- | -------------- | ----------- |
| Data used per update | Entire dataset | One sample  |
| Speed                | Slow           | Fast        |
| Stability            | High           | Low         |
| Memory usage         | High           | Low         |
| Convergence          | Accurate       | Approximate |

---

## ⚖️ Mini-Batch Gradient Descent

A practical combination of both methods:

* Uses small batches of data (e.g., 32, 64 samples)
* Balances speed and stability