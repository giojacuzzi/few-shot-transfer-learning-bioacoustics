import numpy as np

# Sigmoid activation for raw logit confidence values
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid activation pulled directly from BirdNET analyzer.py flat_sigmoid
def sigmoid_BirdNET(x, sensitivity=-1):
    return 1 / (1.0 + np.exp(sensitivity * np.clip(x, -15, 15)))
