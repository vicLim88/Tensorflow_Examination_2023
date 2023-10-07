import tensorflow as tf
from typing import List

import vic_lim_wx as vic


def check_gpu_resources() -> None:
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    logger.info(f"GPUs : {tf.config.list_physical_devices('GPU')}")


def create_tensors_with_tf_constant(tensors: List[List[int]]) -> None:
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    scalar = tf.constant(value=tensors)
    logger.info(scalar.shape)


def tensors_variable() -> None:
    changable_tensor: tf.Variable = tf.Variable([10, 7])
    unchangable_tensor: tf.constant = tf.constant([10, 7])
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini",
                                   class_name="tensors_variable").get_logger()
    logger.info(
        msg=f"""Tensors Variable :
        \t\t-Changable tensor:{changable_tensor}
        \t\t-Unchangable tensor:{unchangable_tensor}
        """
    )


def getting_information_from_tensor(shape_arr: List[int]) -> None:
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    rank_4_tensor = tf.zeros(shape=shape_arr)

    logger.info(
        msg=f"""Tensor's Attribute with shape {shape_arr}
            \t\t Shape              : {rank_4_tensor.shape}
            \t\t Rank               : {rank_4_tensor.ndim}
            \t\t Element DataType   : {rank_4_tensor.dtype}
            \t\t Element Total      : {tf.size(rank_4_tensor).numpy()}
            \t\t Axis_0             : {rank_4_tensor.shape[0]}
            \t\t Axis_1             : {rank_4_tensor.shape[-1]}
            """
    )


if __name__ == "__main__":
    check_gpu_resources()
    # create_tensors_with_tf_constant(tensors=[
    #     [1, 2, 3, 4, 5, 6],
    #     [1, 2, 3, 4, 5, 6]
    # ])
    # tensors_variable()
    getting_information_from_tensor([2,3,4,5])
