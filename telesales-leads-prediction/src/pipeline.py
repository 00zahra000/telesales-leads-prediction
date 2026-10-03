"""Run the complete batch workflow, stopping at the first failed stage."""

import subprocess
import sys


STAGES = [
    ("Connection and source checks", ["-m", "unittest", "-v", "tests.test_connection", "tests.test_source_data"]),
    ("Load data", ["-m", "src.load_data"]),
    ("Loaded data checks", ["-m", "unittest", "-v", "tests.test_loaded_data"]),
    ("Analyze data", ["-m", "src.data_analysis"]),
    ("Train and evaluate", ["-m", "src.train"]),
    ("Score leads", ["-m", "src.predict"]),
    ("Prediction coverage checks", ["-m", "unittest", "-v", "tests.test_prediction_scores"]),
]


def main():
    for label, arguments in STAGES:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, *arguments], check=False)
        if result.returncode:
            print(f"Pipeline stopped: {label} failed (exit {result.returncode}).",
                  file=sys.stderr, flush=True)
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    print("\nPipeline complete: reports, charts, model and lead scores refreshed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
