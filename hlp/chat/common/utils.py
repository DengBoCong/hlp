import os
import logging
import tensorflow as tf
from optparse import OptionParser


class CmdParser(OptionParser):
    def error(self, msg):
        print('Error!提示信息如下：')
        self.print_help()
        self.exit(0)

    def exit(self, status=0, msg=None):
        exit(status)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    优化器将 Adam 优化器与自定义的学习速率调度程序配合使用，这里直接参考了官网的实现
    因为是公式的原因，其实大同小异
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        raise NotImplementedError("Learning rate schedule must override get_config")


def log_operator(level: str, log_file: str = None,
                 log_format: str = "[%(levelname)s] - [%(asctime)s] - [file: %(filename)s] - "
                                   "[function: %(funcName)s] - [%(message)s]") -> logging.Logger:
    """
    日志操作方法，日志级别有'CRITICAL','FATAL','ERROR','WARN','WARNING','INFO','DEBUG','NOTSET'
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    Args:
        log_file: 日志路径
        message: 日志信息
        level: 日志级别
        log_format: 日志信息格式
    Returns:
    """
    if log_file is None:
        log_file = os.path.dirname(__file__)[:-6] + 'data\\runtime.log'

    logger = logging.getLogger()
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level=level)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
