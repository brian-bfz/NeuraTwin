import subprocess
import sys
import os
import time
import warnings

CASE_NAME = "single_push_rope"
NUM_RUNS = 5

# Suppress all warnings in this script
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow, just in case
os.environ["WARP_DISABLE_WARNINGS"] = "1"  # If Warp supports this

for i in range(1, NUM_RUNS + 1):
    print(f"{CASE_NAME} Run {i}")
    # Run generate_data.py with all warnings suppressed
    # Use subprocess to pass -W ignore and env
    subprocess.run([
        sys.executable, "-W", "ignore", "generate_data.py",
        "--case_name", CASE_NAME
    ], env=os.environ.copy())