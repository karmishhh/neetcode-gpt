import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.matmul(x, w) + b
        if activation == 'relu':
            return np.round(self.relu(z),5)
        elif activation == 'sigmoid':
            return np.round(self.sigmoid(z),5) 