import logging

import numpy as np


class ParameterNormalizer:
    def __init__(self, min_vals, max_vals):
        self.min_vals = np.asarray(min_vals, dtype=np.float32)
        self.max_vals = np.asarray(max_vals, dtype=np.float32)
        self.ranges = self.max_vals - self.min_vals

        if np.any(self.ranges <= 0):
            raise ValueError(
                "Each parameter range must satisfy max_vals > min_vals. "
                f"Got min_vals={self.min_vals}, max_vals={self.max_vals}."
            )

    def normalize(self, params):
        return (params - self.min_vals) / self.ranges

    def denormalize(self, params_norm):
        return params_norm * self.ranges + self.min_vals


def setup_file_logger(logger_name, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger
