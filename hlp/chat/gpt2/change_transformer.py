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
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)  # equal 将位置值为0 的变成了1

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 就是得到一个值为1的上三角矩阵
def create_look_ahead_mask(size):
    # tf.linalg.band_part,它的主要功能是以对角线为中心，取它的副对角线部分，其他部分设置为0
    # 参数1:input; 参数2:所要保留的下三角副对角线数量; 参数3:要保留的上三角副对角线数量 负数前部保存
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def creat_mask(inp):
    # 编码器填充遮挡
    # padding_mask = create_padding_mask(inp)

    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）
    look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
    dec_target_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 取对应元素 较大的值

    return combined_mask


def sacled_dot_product_attention(q, k, v, mask):
    """
    计算注意力权重
    q, k, v 必须具备匹配的迁至维度
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
    q: 请求的形状 == (..., seq_len_q, depth)
    k: 主键的形状 == (..., seq_len_k, depth)
    v: 数值的形状 == (..., seq_len_v, depth_v)
    mask: Float 张量，其形状能转换成
          (..., seq_len_q, seq_len_k)。默认为None。

  返回值:
    输出，注意力权重

    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上
    if mask is not None:  # 如果需要mask  将乘以负无穷，如此矩阵为1 的位置 softmax 后变为负无穷，不影响softmax
        scaled_attention_logits += (mask * -1e9)  # mask是padding矩阵

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数
    # 相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):  # dff内部层维数
    '''
    两个全连接网络
     Args:
        d_model:第二层dense的维度
        dff: 第一层dense的维度

    Returns:包含两个dense层的Sequential
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# 多头注意力机制
class MutiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # 本来QKV维度：【batch size，seq.length, embed dim】
        # 拆分后就是【batch size，seq length，h，embed dim/h】
        # depth 就是用embed dim 除头的个数
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # 初始化 q k v 矩阵
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_zise):
        """
        分拆最后一个维度到（num_heads,depth)
        转置结果使得形状为（batch_size,num_heads,seq_len,depth)
        """
        x = tf.reshape(x, (batch_zise, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size,seq_len,d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size,num_heads,seq_len,depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # 之前没有分头的时候是：【batch,size, seq_length, emded_dim】

        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        sacled_attention, attention_weights = sacled_dot_product_attention(q, k, v, mask)

        sacled_attention = tf.transpose(sacled_attention, perm=[0, 2, 1, 3])  # 转置是为了残差连接维度对应
        concat_attention = tf.reshape(sacled_attention,  # 在此处合并多头 得到[batch_size, num_heads, emdeddim]
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights


class block(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(block, self).__init__()

        self.mha = MutiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        """layernorm作用是把神经网络中隐藏层归一为标准正态分布
        也就是i . i . d i.i.di.i.d独立同分布, 以起到加快训练速度, 加速收敛的作用"""

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 加速收敛
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        # 多头注意力
        attn1, attn_weights = self.mha(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # 残差连接

        # 前馈
        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out, attn_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.block = [block(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        attention_weights = {}

        for i in range(self.num_layers):
            x, block = self.block[i](x, training, look_ahead_mask)

            attention_weights['decoder_layer{}_block'.format(i + 1)] = block

        # x.shape == (batch_size, seq_len, d_model)
        return x, attention_weights  # 返回x 注意力权重


class TFGPT2Model(tf.keras.layers.Layer):
    def __init__(self, d_model, vocab_size, max_len, num_heads, dff, num_layers, rate, **kwargs):
        super(TFGPT2Model, self).__init__()
        self.encoder = Encoder(d_model, vocab_size, max_len, rate=0.1)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate=0.1)

        # (batch_size, tar_seq_len, vocab_size)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, training, look_ahead_mask):
        enc_output = self.encoder(inp, training)

        dec_output, attn_weights = self.decoder(enc_output, training, look_ahead_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attn_weights

# inp= tf.constant([[1, 2, 3]])
# tar_inp = inp[:, :-1]
#
# padding_mask, combined_mask = creat_mask(inp)
#
#
# transformer = Transformer(d_model=4, vocab_size=6, max_len=7, num_heads=2, dff=10, num_layers=2)
#
# final_output, attn_weights = transformer(inp, True,combined_mask)
# print('final_output={}, attn_weights={}'.format(final_output, attn_weights))
