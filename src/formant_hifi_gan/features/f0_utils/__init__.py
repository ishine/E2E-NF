import numpy as np

from .extractor import F0_Extractor

__all__ = [
    "F0_Extractor",
    "logf0",
    "log2f0",
    "vuv",
    "sumlf0",
    "calc_f0_mean",
]


def logf0(f0: np.ndarray):
    lf0 = f0.copy()
    nonzero_indices = np.nonzero(lf0)
    lf0[nonzero_indices] = np.log(lf0[nonzero_indices])

    return lf0


# octave [cent] = 1200 * log2f0(f0 / base_freq)
def log2f0(f0: np.ndarray) -> np.ndarray:
    log2f0 = f0.copy()
    nonzero_indices = np.nonzero(log2f0)
    log2f0[nonzero_indices] = np.log2(log2f0[nonzero_indices])

    return log2f0


def vuv(f0: np.ndarray) -> np.ndarray:
    nonzero_indices = np.nonzero(f0)
    vuv = np.zeros_like(f0)
    vuv[nonzero_indices] = 1

    return vuv


def sumlf0(lf0: np.ndarray) -> float:
    return np.sum(lf0[lf0 != 0.0])


def calc_f0_mean(f0_and_sn: tuple[np.ndarray, np.ndarray]) -> float:
    lf0_mean = 0.0
    n_v = 0
    for f0, sn in f0_and_sn:
        lf0_mean += sumlf0(logf0(f0) * sn)
        n_v += np.sum(vuv(f0 * sn))
    lf0_mean /= n_v
    f0_mean = np.exp(lf0_mean)

    return f0_mean
