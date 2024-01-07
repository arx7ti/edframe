import numpy as np
import itertools as it
import unittest as test

from edframe.data.entities2 import VI
from edframe.data.generators import make_hf_cycles

F0 = [40, 45, 50, 55, 60]
FS = [1000, 2000, 4000]
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)


def init_data():
    X = []
    params = it.product(F0, FS)

    for f0, fs in params:
        dt = rng.uniform(1 / f0, 10 / f0)
        I, _ = make_hf_cycles(1, 1, fs=fs, f0=f0)
        i = I[0]
        i /= abs(i).max()
        t = np.linspace(0, dt, len(i))
        v = np.sin(2 * np.pi * f0 * t)

        x = VI(v, i, fs, f0=f0, is_aligned=True)
        X.append(x)

    return X


class TestVI(test.TestCase):

    X = init_data()

    def test_components(self):
        pairs = []

        for k in range(2, 10):
            ids = rng.choice(len(self.X), size=k)
            Xk = [self.X[i] for i in ids]
            # TODO


if __name__ == '__main__':
    test.main()
