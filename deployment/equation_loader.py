import ctypes
import os
import numpy as np

_lib_path = os.path.join(os.path.dirname(__file__), "libequation.so")
_lib = None
if os.path.exists(_lib_path):
    try:
        _lib = ctypes.CDLL(_lib_path)
        _lib.evaluate_residual.argtypes = [ctypes.POINTER(ctypes.c_double)]
        _lib.evaluate_residual.restype = ctypes.c_double
        _lib.evaluate_residual_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int
        ]
        _lib.evaluate_residual_batch.restype = None
    except Exception as e:
        print(f"Warning: Failed to load libequation.so: {e}")
        _lib = None


def evaluate_residual(x):
    if _lib is not None:
        arr = np.ascontiguousarray(x, dtype=np.float64)
        return _lib.evaluate_residual(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    # Fallback to pure Python
    val = -x[1]*x[3] + x[1] + 3*x[0] - 5*x[4] - log(fabs(x[2]) + 1.0e-6) - sqrt(fabs(x[4] + 0.492))
    return float(np.clip(val, -30.0, 30.0))


def evaluate_residual_batch(X):
    n = len(X)
    if _lib is not None:
        arr = np.ascontiguousarray(X, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        _lib.evaluate_residual_batch(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n
        )
        return out
    return np.array([evaluate_residual(x) for x in X])
