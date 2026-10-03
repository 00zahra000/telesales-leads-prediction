"""Run the complete batch workflow, stopping at the first failed stage."""

import subprocess
import sys

from src.logging_config import logger


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
        logger.info("Starting stage: {}", label)
        result = subprocess.run([sys.executable, *arguments], check=False)
        if result.returncode:
            logger.error("Pipeline stopped: {} failed (exit {}).", label, result.returncode)
            return result.returncode if result.returncode > 0 else 128 - result.returncode
        logger.success("Completed stage: {}", label)
    logger.success("Pipeline complete: reports, charts, model and lead scores refreshed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
