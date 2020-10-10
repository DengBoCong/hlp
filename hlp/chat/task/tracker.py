import tensorflow as tf


class InformSlotTracker(tf.keras.layers.Layer):
    """
    informable插槽跟踪器，informable插槽是用户告知系统的信息，用
    来约束对话的一些条件，系统为了完成任务必须满足这些条件
    用来获得时间t的状态的槽值分布，比如price=cheap
    输入为状态跟踪器的输入'state_t'，输出为槽值分布'P(v_s_t| state_t)'
    """

    def __init__(self, n_choices):
        super(InformSlotTracker, self).__init__()
        self.n_choices = n_choices + 1 # 因为包括dontcare
        self.fc = tf.keras.layers.Dense(units=n_choices)

    def call(self, state):
        return self.fc(state)


class RequestSlotTracker(tf.keras.layers.Layer):
    """
    requestable插槽跟踪器，requestable插槽是用户询问系统的信息
    用来获得时间t的状态的非分类插槽槽值分布，
    比如：
    address=1 (地址被询问)
    phone=0 (用户不关心电话号码)
    输入为状态跟踪器的输入'state_t'，输出为槽值二元分布'P(v_s_t| state_t)'
    """

    def __init__(self):
        super(RequestSlotTracker, self).__init__()
        self.fc = tf.keras.layers.Dense(2)

    def call(self, state):
        return self.fc(state)
