import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.matmul(w,x) + b
        y_pred = 1 / (1+ np.exp(-z))
        # loss = np.mean((y_pred - y_true)**2)
        dloss_dypred = (y_pred-y_true)
        dypred_dz = y_pred * (1-y_pred)
        dz_dw = x 
        dz_db = 1
        dl_dw = dloss_dypred * dypred_dz * dz_dw 
        dl_db = dloss_dypred * dypred_dz * dz_db

        return np.round(dl_dw,5) , np.round(dl_db,5)

