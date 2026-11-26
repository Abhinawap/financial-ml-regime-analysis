import os

# XGBoost threads oversubscribe a shared machine: a full run goes from ~30 s to 20+ min.
os.environ.setdefault("OMP_NUM_THREADS", "1")
