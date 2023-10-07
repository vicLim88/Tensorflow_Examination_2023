import sys

import tensorflow as tf
import vic_lim_wx as vic

logger = vic.Vic_Custom_Logger(
    config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()


def check_python_version() -> None:
    logger.info(
        msg=f"""Python
            Version     : {sys.version}
        """)


def check_tensorflow_version() -> None:
    logger.info(
        msg=f"""Tensorflow
            Version     : {tf.__version__}
            GPU Device  : {tf.config.list_physical_devices('GPU')}
        """)


if __name__ == "__main__":
    check_python_version()
    check_tensorflow_version()
