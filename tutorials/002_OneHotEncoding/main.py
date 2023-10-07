import tensorflow as tf
from typing import List

import vic_lim_wx as vic


def check_gpu_resources() -> None:
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    logger.info(f"GPUs : {tf.config.list_physical_devices()}")


def one_hot_encode(cat_cols_converted: List[int]) -> None:
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()

    logger.info(f"One Hot :\n{tf.one_hot(indices=cat_cols_converted, depth=len(cat_cols_converted))}")


def convert_cols_to_one_hot_compatible(cat_cols_raw: List[str]):
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    color_indices: List[int] = tf.range(len(cat_cols_raw))

    logger.info(f"Converted indices to tensors\n{color_indices}")
    return color_indices


if __name__ == "__main__":
    check_gpu_resources()
    # one_hot_encode([1, 2, 3, 4])
    # cols_converted: List[int] = convert_cols_to_one_hot_compatible(["red", "blue", "green", "yellow"])
    # one_hot_encode(cols_converted)
