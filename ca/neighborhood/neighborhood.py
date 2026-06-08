import numpy as np

class Neighborhood:
    def __init__(self, kernel: np.ndarray, boundary: str = 'wrap'):
        self.kernel = kernel
        self.boundary = boundary

    # Defaults:
    Moore: 'Neighborhood' = None
    Neumann: 'Neighborhood' = None
    
    @classmethod
    def Custom(cls, kernel: np.ndarray, boundary: str = 'wrap') -> 'Neighborhood':
        return cls(kernel, boundary)
    
Neighborhood.Moore = Neighborhood(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]))
Neighborhood.Neumann = Neighborhood(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))