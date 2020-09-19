import os
import jieba
import tensorflow as tf
from pathlib import Path
import common.data_utils as _data
import config.get_config as _config


class Chatter(object):
    """"
    面向使用者的聊天器基类
    该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
    不同模型或方法实现的聊天子类化该类。
    """

    def __init__(self, checkpoint_dir):
        """
        Transformer聊天器初始化，用于加载模型
        """
        _, self.input_token, _, self.target_token = _data.load_dataset()
        is_exist = Path(checkpoint_dir)
        if not is_exist.exists():
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt = tf.io.gfile.listdir(checkpoint_dir)

    def respond(self, req):
        """ 对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        :param req: 外部聊天请求字符串
        :return: 系统回复字符串
        """
        pass

    def stop(self):
        """ 结束聊天

        可以做一些清理工作
        :return:
        """
        pass

    def pre_treat_inputs(self, sentence):
        # 分词
        sentence = " ".join(jieba.cut(sentence))
        # 添加首尾符号
        sentence = _data.preprocess_sentence(sentence)
        # 将句子转成token列表
        inputs = [self.input_token.word_index.get(i, 3) for i in sentence.split(' ')]
        # 填充
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
        # 转成Tensor
        inputs = tf.convert_to_tensor(inputs)
        # decoder的input就是开始符号
        dec_input = tf.expand_dims([self.target_token.word_index['start']], 0)
        return inputs, dec_input

    # def treat_predictions(self, predictions):
    #     # 取概率最大的值
    #     predicted_id = tf.cast(tf.argmax(predictions), tf.int32).numpy()
    #     # 返回对应token和最大的值
    #     return self.target_token.word_index.get(predicted_id, 'end'), predicted_id
