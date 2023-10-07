import tensorflow as tf


def check_gpu_resources() -> None:
    print(f"GPUs : {tf.config.list_physical_devices('GPU')}")


if __name__ == "__main__":
    check_gpu_resources()
    print("Hello")
