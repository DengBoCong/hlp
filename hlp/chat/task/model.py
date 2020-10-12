import json
import tensorflow as tf
import config.get_config as _config
from task.tracker import InformSlotTracker, RequestSlotTracker


def gen_tracker_model(onto, kb):
    """
    根据inform和request的槽位的个数，生成对应的tracker
    :param onto: 处理过的本体数据集
    :param state_tracker_hidden_size: 处理过的本体数据集
    :param kb: 处理过的本体数据集
    """
    slot_trackers = {}
    slot_len_sum = 0

    for slot in onto:
        if len(onto[slot]) > 2:
            slot_trackers[slot] = InformSlotTracker(len(onto[slot]))
            slot_len_sum += len(onto[slot])
        else:
            slot_trackers[slot] = RequestSlotTracker()
            slot_len_sum += 2

    return slot_len_sum


def encoder(units, vocab_size, embedding_dim, output_dim, att=False, name="task_encoder"):
    inputs = tf.keras.Input(shape=(None,), name='task_encoder_inputs')
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)

    forward_lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True)
    backword_lstm = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True, go_backwards=True)
    output, state = tf.keras.layers.Bidirectional(layer=forward_lstm, backward_layer=backword_lstm, merge_mode="concat")(embedding)




def task(vocab_size):
    """
    Task-Orient模型，使用函数式API实现，将encoder和decoder封装
    :param vocab_size:token大小
    """
