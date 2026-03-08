from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-12


def dbm_to_watt(dbm: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** ((np.asarray(dbm) - 30.0) / 10.0)


def watt_to_dbm(watt: float | np.ndarray) -> float | np.ndarray:
    watt_arr = np.maximum(np.asarray(watt), EPS)
    return 10.0 * np.log10(watt_arr) + 30.0


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    return 10.0 ** (np.asarray(db) / 10.0)


def linear_to_db(value: float | np.ndarray) -> float | np.ndarray:
    value_arr = np.maximum(np.asarray(value), EPS)
    return 10.0 * np.log10(value_arr)


def complex_normal(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= EPS:
        return np.zeros_like(v)
    return v / norm


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    x = np.asarray(weights, dtype=float)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= EPS:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())
