import subprocess
import sys
import os

def debug_hwp():
    cmd = [sys.executable, "-m", "hwp5.hwp5html", "--output", "debug_output", "test.hwp"]
    print(f"Running: {' '.join(cmd)}")
    
    env = os.environ.copy()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    print(f"Return Code: {result.returncode}")

if __name__ == "__main__":
    debug_hwp()
