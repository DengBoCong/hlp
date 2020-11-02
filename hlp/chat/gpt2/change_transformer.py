import tensorflow as tf
import numpy as np

# 位置编码函数


"""位置编码公式"""


# 位置编码----每个位置都得到d_model维的位置向量
def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates  # 维度（position，d_model)每个位置都得到d_model维的位置向量


def positional_encoding(position, d_model):
    '''位置编码，在embedding中加入位置信息'''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 从0开始到末，步长为2

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]  # np.newaxis 增多一维

    return tf.cast(pos_encoding, dtype=tf.float32)  # 维度 （1，position,d_model)


class Encoder(tf.keras.layers.Layer):
    """
    包含
    - 输入嵌入（Input Embedding）
    - 位置编码（Positional Encoding）
    输入经过嵌入（embedding）后，该嵌入与位置编码相加。该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入。

    """

    def __init__(self, d_model, vocab_size, max_len, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        return x


# class TFGPT2Model(tf.keras.Model):
#     def __init__(self,  *inputs, **kwargs):
#         super().__init__( *inputs, **kwargs)
#         self.transformer = Transformer( name="transformer")
#
#     def call(self, inputs, **kwargs):
#         outputs = self.transformer(inputs, **kwargs)
#         return outputs

# 填充遮挡
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 就是得到一个值为1的上三角矩阵
def create_look_ahead_mask(size):
    # tf.linalg.band_part,它的主要功能是以对角线为中心，取它的副对角线部分，其他部分设置为0
    # 参数1:input; 参数2:所要保留的下三角副对角线数量; 参数3:要保留的上三角副对角线数量 负数前部保存
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def creat_mask(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 取对应元素 较大的值

    return enc_padding_mask, combined_mask


class Transformer(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, max_len, **kwargs):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, vocab_size, max_len, rate=0.1)

    def call(self, inp, training):
        enc_output = self.encoder(inp, training)
        print(enc_output)


transformer = Transformer(d_model=5, vocab_size=6, max_len=7)
x = tf.constant([[1, 2, 3]])
transformer(x, True)
