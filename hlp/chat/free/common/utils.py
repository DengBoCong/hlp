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
