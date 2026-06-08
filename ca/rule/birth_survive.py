import numpy as np

from scipy.signal import convolve2d

from .rule import Rule

from ca.neighborhood import Neighborhood

type BirthSurviveParam = int | range | set[int] | list[int]

def _to_frozenset(param: BirthSurviveParam):
    if isinstance(param, int):
        return frozenset({param})
    
    return frozenset(param)

class BirthSurvive(Rule):
    def __init__(self, birth: BirthSurviveParam, survive: BirthSurviveParam, neighborhood: Neighborhood = Neighborhood.Moore):
        self._birth = _to_frozenset(birth)
        self._survive = _to_frozenset(survive)

        self._neighborhood = neighborhood

    def apply(self, grid: np.ndarray) -> np.ndarray:
        neighbors = convolve2d(grid, self._neighborhood.kernel, mode='same', boundary=self._neighborhood.boundary)

        birth = (grid == 0) & np.isin(neighbors, list(self._birth))
        survive = (grid == 1) & np.isin(neighbors, list(self._survive))

        return (birth | survive).view(np.uint8)
    
    # Defaults:
    GOL: 'BirthSurvive' = None
    B34S23: 'BirthSurvive' = None
    B35S23: 'BirthSurvive' = None

    @classmethod
    def Custom(cls, birth: BirthSurviveParam, survive: BirthSurviveParam, neighborhood: Neighborhood = Neighborhood.Moore) -> 'BirthSurvive':
        return cls(birth, survive, neighborhood)

BirthSurvive.GOL = BirthSurvive(birth=3, survive=range(2, 3))
BirthSurvive.B34S23 = BirthSurvive(birth=range(3, 5), survive=range(2, 4))
BirthSurvive.B35S23 = BirthSurvive(birth=range(3, 6), survive=range(2, 4))