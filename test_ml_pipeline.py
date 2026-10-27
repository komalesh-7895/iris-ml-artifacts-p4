import json
import subprocess
import sys

def test_quality_gate_passes():
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": 0.90}, f)

    result = subprocess.run(
        [sys.executable, "quality_gate.py"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "QUALITY GATE PASSED" in result.stdout
