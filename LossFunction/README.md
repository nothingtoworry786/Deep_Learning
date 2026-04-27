# Loss Functions in Deep Learning

## What is a Loss Function?

A loss function measures how far your model’s prediction is from the actual value.

* High loss → poor prediction
* Low loss → good prediction

You can think of it as a score that tells the model how wrong it is.

---

## Why is it Important?

A machine learning model improves by minimizing the loss.

During training:

* The model makes predictions
* The loss function calculates error
* The optimizer updates the model to reduce that error

Without a loss function, the model has no way to learn.

---

## Loss vs Cost Function

* Loss Function → error for a single data point
* Cost Function → average of all losses in the dataset

In practice, we usually optimize the cost function.

---

## How to Choose the Right Loss Function

Choosing the correct loss function is critical.

| Problem Type               | Recommended Loss Function |
| -------------------------- | ------------------------- |
| Regression                 | MSE, MAE, Huber           |
| Binary Classification      | Binary Cross Entropy      |
| Multi-class Classification | Categorical Cross Entropy |
| SVM                        | Hinge Loss                |
| Generative Models          | KL Divergence, GAN Loss   |
| Embeddings                 | Triplet Loss              |

If you choose the wrong loss function, the model may not learn correctly.

---

# Loss Functions

---

## 1. Mean Squared Error (MSE)

### Explanation

Measures the squared difference between actual and predicted values.
Larger errors are penalized more heavily.

Used in regression problems.

### Formula

```
Loss = (yi - ŷi)^2  
Cost = Σ(yi - ŷi)^2 / n
```

### Advantages

* Simple and easy to implement
* Differentiable
* Works well when there are no outliers

### Disadvantages

* Very sensitive to outliers
* Error is squared, so units are harder to interpret

---

## 2. Mean Absolute Error (MAE)

### Explanation

Measures the absolute difference between actual and predicted values.

Used in regression when outliers are present.

### Formula

```
Loss = |yi - ŷi|  
Cost = Σ|yi - ŷi| / n
```

### Advantages

* Easy to understand
* Same unit as original data
* Robust to outliers

### Disadvantages

* Not differentiable at zero
* Can lead to slower convergence

---

## 3. Huber Loss

### Explanation

A combination of MSE and MAE.

* Small errors → behaves like MSE
* Large errors → behaves like MAE

Useful when data contains some outliers.

### Formula

```
if |error| <= δ:
    0.5 * error^2
else:
    δ * (|error| - 0.5 * δ)
```

### Advantages

* Less sensitive to outliers than MSE
* Smooth and differentiable

### Disadvantages

* Requires choosing a delta (δ) value
* Slightly more complex

---

## 4. Binary Cross Entropy (BCE)

### Explanation

Used for binary classification problems (output is 0 or 1).

Examples: spam detection, fraud detection.

### Formula

```
Loss = -[y log(p) + (1 - y) log(1 - p)]
```

### Advantages

* Works well with probability outputs
* Standard choice for binary classification

### Disadvantages

* Strong penalty for confident wrong predictions

---

## 5. Categorical Cross Entropy (CCE)

### Explanation

Used for multi-class classification problems.

Examples: digit classification, image labeling.

### Formula

```
Loss = - Σ yi log(pi)
```

### Advantages

* Works well with softmax
* Effective for multi-class problems

### Disadvantages

* Requires proper probability distribution

---

## 6. Hinge Loss

### Explanation

Used in Support Vector Machines (SVM).
Focuses on maximizing the margin between classes.

### Formula

```
Loss = max(0, 1 - y * y_pred)
```

### Advantages

* Good for margin-based classification
* Helps improve class separation

### Disadvantages

* Not smooth
* Harder to optimize

---

## 7. KL Divergence

### Explanation

Measures how one probability distribution differs from another.

Used in autoencoders and generative models.

### Formula

```
KL = Σ P(x) log(P(x) / Q(x))
```

### Advantages

* Useful in probabilistic models
* Measures distribution similarity

### Disadvantages

* Not symmetric
* Can be unstable

---

## 8. GAN Loss

### Explanation

Used in Generative Adversarial Networks.

Two components:

* Discriminator → distinguishes real vs fake data
* Generator → tries to produce realistic data

### Idea

* Generator minimizes loss
* Discriminator maximizes accuracy

### Advantages

* Can generate realistic data

### Disadvantages

* Difficult to train
* Often unstable

---

## 9. Triplet Loss

### Explanation

Used in embedding learning (e.g., face recognition).

It uses three inputs:

* Anchor
* Positive (same class)
* Negative (different class)

### Formula

```
Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

### Advantages

* Learns similarity effectively
* Useful for embeddings

### Disadvantages

* Requires careful data selection
* Training can be slow
