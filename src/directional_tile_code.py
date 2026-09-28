import numpy as np

from .directional_word import DirectionalWord


class DirTileCode:
    def __init__(self, M: int, N: int, word: DirectionalWord):
        self.M = M
        self.N = N
        self.word = word

        self.B = word.B
        self.padding = self.B - 1
        self.anchor_shape = (self.M + 2 * self.padding, self.N + 2 * self.padding)

    def empty_anchor_mask(self):
        return np.zeros(self.anchor_shape, dtype=bool)

    @property
    def bulk_mask(self):
        p = self.padding

        mask = self._empty_anchor_mask()
        mask[p:p + self.M, p:p + self.N] = True

        return mask

    @property
    def x_boundary_mask(self):
        p = self.padding
        mask = self._empty_anchor_mask()

        mask[p:p + self.M, :p] = True
        mask[p:p + self.M, p + self.N:] = True

        return mask

    @property
    def z_boundary_mask(self):
        p = self.padding
        mask = self._empty_anchor_mask()

        mask[:p, p:p + self.N] = True
        mask[p + self.M:, p:p + self.N] = True

        return mask

    @property
    def x_anchor_mask(self):
        return self.bulk_mask | self.x_boundary_mask

    @property
    def z_anchor_mask(self):
        return self.bulk_mask | self.z_boundary_mask
