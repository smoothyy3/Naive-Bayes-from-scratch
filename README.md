# Naive-Bayes-from-scratch
This project demonstrates a simple implementation of the Naive Bayes Classifier from scratch in Python.

## Description
The **Naive Bayes Classifier** calculates the posterior probability of each class given the input data and predicts the class with the highest probability. The classifier assumes:
- All features are independent (conditional independence).
- The distribution of features follows a known probability distribution (e.g., Gaussian, Bernoulli, or Multinomial).

This implementation is built without any imports, relying only on Python's basic functionality to handle mathematical operations and data manipulation.

---

## Features
- **Pure Python Implementation**: No external libraries or helper functions are used.
- **Multiclass Classification**: Supports multiple target classes.
- **Flexible Probability Distributions**: The implementation will initially support the Gaussian distribution, with potential extensions to Bernoulli and Multinomial distributions.

---

## To-Do
- ❌ Implement prior probability calculation \( P(C) \).
- ❌ Implement likelihood calculation for Gaussian distribution \( P(x|C) \).
- ❌ Implement posterior probability calculation \( P(C|x) \) using Bayes' Theorem.
- ❌ Add prediction functionality for single and batch inputs.
- ❌ Add example dataset (e.g., Iris or custom).
- ❌ Implement accuracy evaluation for the classifier.
- ❌ Extend support to Bernoulli and Multinomial distributions.

---
