import subprocess
import sys
import os
import re
import warnings
from datetime import datetime

CASE_NAME = "single_push_rope"
START_TIME = "20250521_133000"  # Change as needed
DATA_DIR = "generated_data"

# Suppress all warnings in this script
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WARP_DISABLE_WARNINGS"] = "1"

def parse_timestamp(ts):
    # Accepts formats like YYYYMMDD_HHMMSS or YYYYMMDDHHMMSS
    ts = ts.strip('_')
    m = re.match(r"(\d{8})[_-]?(\d{6})", ts)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return None

start_dt = parse_timestamp(START_TIME)
if start_dt is None:
    print(f"Could not parse START_TIME: {START_TIME}")
    sys.exit(1)

for entry in os.listdir(DATA_DIR):
    if not entry.startswith(f"{CASE_NAME}_"):
        continue
    ts_part = entry[len(CASE_NAME)+1:]
    ts_dt = parse_timestamp(ts_part)
    print(f"<{ts_part}> <{START_TIME}> parsed: {ts_dt}")
    if ts_dt and ts_dt >= start_dt:
        print(f"Rendering video for {entry} (timestamp: {ts_part})")
        subprocess.run([
            sys.executable, "-W", "ignore", "v_from_d.py",
            "--case_name", CASE_NAME,
            "--timestamp", ts_part
        ], env=os.environ.copy()) 