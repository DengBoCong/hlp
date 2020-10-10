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
            slot_len_sum += len(onto[slot]) + 1
        else:
            slot_trackers[slot] = RequestSlotTracker()
            slot_len_sum += 2

def encoder(vocab_size, embedding_dim, hidden_size, output_dim, att=False):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    output, hidden = tf.keras.layers.GRU()(embedding)


def task(vocab_size):
    """
    Task-Orient模型，使用函数式API实现，将encoder和decoder封装
    :param vocab_size:token大小
    """