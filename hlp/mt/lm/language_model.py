import os

from pathlib import Path
import tensorflow as tf

from hlp.mt.config import get_config as _config


class LanguageModel(tf.keras.Model):
    """
    语言模型，将input输入
    """

    def __init__(self, vocab_size, d_embedding, batch_size, d_rnn):
        super(LanguageModel, self).__init__()
        # 初始参数
        self.batch_size = batch_size
        self.d_rnn = d_rnn
        self.d_embedding = d_embedding
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, d_embedding)
        self.state0 = [tf.zeros([batch_size, d_rnn]), tf.zeros([batch_size, d_rnn])]
        self.state1 = [tf.zeros([batch_size, d_rnn]), tf.zeros([batch_size, d_rnn])]

        self.cell0 = tf.keras.layers.LSTMCell(d_rnn)
        self.cell1 = tf.keras.layers.LSTMCell(d_rnn)

        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, sequences):
        """
        传入已编码的句子,shape ---> (batch_size, seq_len)
        返回预测序列
        """
        output = []
        sequences = self.embedding(sequences)  # shape ---> (batch_size, seq_len, vocab_size)
        sequences *= tf.math.sqrt(tf.cast(self.d_embedding, tf.float32))
        # output 为输出的字符列表，每个列表元素shape --> (batch_size, vocab_size)
        for sequences_t in tf.unstack(sequences, axis=1):  # sequences_t.shape --> (batch_size, vocab_size)
            out0, self.state0 = self.cell0(sequences_t, self.state0)  # out0.shape --> (batch_size, vocab_size)
            out1, self.state1 = self.cell1(out0, self.state1)
            out1 = self.output_layer(out1)
            output.append(out1)

        predictions = tf.stack(output, axis=1)  # prediction.shape --> (batch_size, seq_len, vocab_size)
        return predictions

    def reset_states(self):
        super(LanguageModel, self).reset_states()
        self.state0 = [tf.zeros([self.batch_size, self.d_rnn]), tf.zeros([self.batch_size, self.d_rnn])]
        self.state1 = [tf.zeros([self.batch_size, self.d_rnn]), tf.zeros([self.batch_size, self.d_rnn])]


def check_point():
    """
    检测检查点目录下是否有文件
    """
    # 进行语言对判断从而确定检查点路径
    checkpoint_dir = _config.lm_checkpoint_path
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    if_ckpt = tf.io.gfile.listdir(checkpoint_dir)
    return if_ckpt

