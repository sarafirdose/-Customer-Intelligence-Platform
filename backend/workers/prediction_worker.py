"""
Prediction Worker Process Entrypoint.

Runs standalone or as a background service to process queued prediction tasks.
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core.logger import logger
from backend.workers.queue_manager import queue_manager


def main() -> None:
    logger.info("Prediction Worker process initialized.")
    logger.info("Monitoring queue for incoming tasks...")

    try:
        while True:
            stats = queue_manager.get_stats()
            if stats["queued"] > 0 or stats["processing"] > 0:
                logger.info(
                    f"Worker active: queued={stats['queued']}, "
                    f"processing={stats['processing']}, completed={stats['completed']}"
                )
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Prediction Worker process shutting down.")


if __name__ == "__main__":
    main()
