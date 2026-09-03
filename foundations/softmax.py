import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.round((np.exp(z - max(z))) / ( sum(np.exp(z - max(z)))), 4)



