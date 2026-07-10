import os


# librosa imports Numba-decorated helpers; disabling JIT keeps tests portable on
# Python versions/environments where Numba cannot cache package-installed code.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
