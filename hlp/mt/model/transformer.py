import tensorflow as tf

from hlp.utils import layers


# 点式前馈网络（Point wise feed forward network）
def point_wise_feed_forward_network(d_model, dff):
    """
    简单的两个全连接层网络
    Args:
        d_model:第二层dense的维度
        dff: 第一层dense的维度

    Returns:包含两个dense层的Sequential

    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# 编码器层（Encoder layer）
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = layers.MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# 解码器层
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(d_model, num_heads)
        self.mha2 = layers.MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = layers.create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = layers.create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = layers.create_look_ahead_mask(tar)
    dec_target_padding_mask = layers.create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# 编码器
class Encoder(tf.keras.layers.Layer):
    """
    包含
    - 输入嵌入（Input Embedding）
    - 位置编码（Positional Encoding）
    - N 个编码器层（encoder layers）
    输入经过嵌入（embedding）后，该嵌入与位置编码相加。该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入。

    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.positional_encoding(maximum_position_encoding,
                                                       self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # 将嵌入和位置编码相加。
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# 解码器（Decoder）
class Decoder(tf.keras.layers.Layer):
    """
    解码器包括：
    - 输出嵌入（Output Embedding）
    - 位置编码（Positional Encoding）
    - N 个解码器层（decoder layers）
    目标（target）经过一个嵌入后，该嵌入和位置编码相加。该加法结果是解码器层的输入。解码器的输出是最后的线性层的输入。
    """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = layers.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# Transformer模型
class Transformer(tf.keras.Model):
    """
    Transformer 包括编码器，解码器和最后的线性层。解码器的输出是线性层的输入，返回线性层的输出。
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# 使用schedual sampling的transformer
class ScheduledSamplingTransformer(tf.keras.Model):
    """
    使用scheduled sampling的Transformer模型
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1, schedule_type=None, temperature=1.0):
        super(Transformer, self).__init__()

        self.read_probability = 0.7
        self.schedule_type = 'constant'
        self.k = None

        self.schedule_type = schedule_type
        self.temperature = temperature

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, step):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_first_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        first_final_output = self.final_layer(dec_first_output)

        gumbel_dec_first_output = self._gumbel_softmax(first_final_output)  # 对第一个decoder输出做高斯模糊

        mix_output = self.embedding_mix(gumbel_dec_first_output, tar)

        dec_second_outputs = self.decoder(mix_output, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_second_outputs)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def _embedding_mix(self, gumbel_inputs, inputs, step):
        """返回schedule sampling的embedding

        :param gumbel_inputs:第一次decoder输出后添加噪音的Output
        :param inputs:真实的target
        :param step:当前步数
        :return:schedule sampling的input
        """
        random = tf.random.uniform(shape=tf.shape(inputs), maxval=1, minval=0, dtype=tf.float32)
        sampling_probability = self._get_sampling_probability(step, read_probability=None, schedule_type=None, k=None)
        return tf.where(random < sampling_probability, x=gumbel_inputs, y=inputs)

    def _gumbel_softmax(self, inputs):
        """
        按照论文中的公式，实现GumbelSoftmax，具体见论文公式
        :param inputs: 输入
        :return: 混合Gumbel噪音后，做softmax以及argmax之后的输出
        """
        uniform = tf.random.uniform(shape=tf.shape(inputs), maxval=1, minval=0)
        # 以给定输入的形状采样Gumbel噪声
        gumbel_noise = -tf.math.log(-tf.math.log(uniform))
        # 将Gumbel噪声添加到输入中，输入第三维就是分数
        gumbel_outputs = inputs + gumbel_noise
        gumbel_outputs = tf.cast(gumbel_outputs, dtype=tf.float32)
        # 在给定温度下，进行softmax并返回
        gumbel_outputs = tf.nn.softmax(self.temperature * gumbel_outputs)
        gumbel_outputs = tf.argmax(gumbel_outputs, axis=-1)
        return tf.cast(gumbel_outputs, dtype=tf.float32)

    def set_decay_config(self, read_probability=None, schedule_type=None, k=None):
        """设置几率衰减的类型及参数

        :param read_probability: 若为线性衰减，初始的几率
        :param schedule_type: 衰减类型: "constant", "linear", "exponential", "inverse_sigmoid"
        :param k: 函数参数
        """
        self.read_probability = read_probability
        self.schedule_type = schedule_type
        self.k = k

    def _get_sampling_probability(self, step):
        """返回几率
        需先使用方法set_decay_config来设置衰减函数类型及
        :param step:当前训练步数
        :return: 几率
        """
        read_probability = self.read_probability
        schedule_type = self.schedule_type
        k = self.k

        if read_probability is None and schedule_type is None:
            return None

        if schedule_type is not None and schedule_type != "constant":
            if k is None:
                raise ValueError("scheduled_sampling_type确定后必须设置scheduled_sampling_k ")

            step = tf.cast(step, tf.float32)
            k = tf.constant(k, tf.float32)

            if schedule_type == "linear":
                if read_probability is None:
                    raise ValueError("Linear schedule 需要一个初始概率")
                read_probability = min(read_probability, 1.0)
                read_probability = tf.maximum(read_probability - k * step, 0.0)
            elif schedule_type == "exponential":
                read_probability = tf.pow(k, step)
            elif schedule_type == "inverse_sigmoid":
                read_probability = k / (k + tf.exp(step / k))
            else:
                raise TypeError("未知的schedule type: {}".format(schedule_type))

        return 1.0 - read_probability
