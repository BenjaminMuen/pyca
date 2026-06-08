import base64
import numpy as np

from .rule import Rule

class Generator:
    def __init__(self, rule: Rule, size: tuple[int, int] = (256, 256), seed: np.ndarray = None):
        self._rule = rule
        
        if seed is None:
            self._grid = np.random.randint(2, size=size, dtype=np.uint8)
            self._size = size
        else:
            self._grid = seed
            self._size = seed.shape

        self.epoch = 0

    @classmethod
    def from_b64(cls, rule: Rule, data: dict[str, int | str]) -> 'Generator':
        h, w = (data['height'], data['width'])

        bytes = base64.b64decode(data['data'].encode('ascii'))

        flat_grid = np.unpackbits(np.frombuffer(bytes, dtype=np.uint8))

        grid = flat_grid[:h * w].reshape(h, w)

        return cls(rule, seed=grid)

    @property
    def b64(self) -> dict[str, int | str]:
        bytes = np.packbits(self._grid)

        b64_string = base64.b64encode(bytes).decode('ascii')

        return {'height': self._size[0], 'width': self._size[1], 'data': b64_string}
    
    def step(self) -> np.ndarray:
        self._grid = self._rule.apply(self._grid)

        self.epoch += 1

        return self._grid
    
    def evolve(self, n: int = 1) -> np.ndarray:
        for _ in range(n):
            self.step()

        return self._grid