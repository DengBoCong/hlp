import tensorflow as tf


def scaled_dot_product_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor):
    """
    这里直接根据注意力的公式进行编写
    Args:
        query: Q
        key: K
        value: V
        mask: 注意力机制的权重
    Returns:
    """
    # 先将query和可以做点积
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    deep = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(deep)

    # mask的作用就是为了将输入中，很大的负数单元在softmax之后归零
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 将最后一个轴上的值进行归一化得到的就是attention score，
    # 然后和value做点积之后，得到我们想要的加权向量
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    按照多头注意力的结构进行编写
    """

    def __init__(self, d_model: int, num_heads: int, name: str = "multi_head_attention"):
        """
        Args:
            d_model: 深度，词嵌入维度
            num_heads: 注意力头数量
            name: 名称
        Returns:
        """
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, inputs: tf.Tensor, batch_size: int):
        """
        Args:
            inputs: 输入
            batch_size: batch大小
        Returns:
        """
        inputs = tf.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs: tf.Tensor):
        """
        Args:
            inputs: 输入
        Returns:
        """
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output


class PositionalEncoding(tf.keras.layers.Layer):
    """
    位置编码的简单实现，实现了位置编码的两个公式(针对奇偶位置进行的编码)
    位置编码原理自行翻阅资料，这边不做注释
    """

    def __init__(self, position: int, d_model: int):
        """
        Args:
            position: 词汇量大小
            d_model: 深度，词嵌入维度
        Returns:
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, d_model: int):
        """
        Args:
            position: 奇偶位置
            i: 奇偶位置
            d_model: 深度，词嵌入维度
        Returns:
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position: int, d_model: int):
        """
        Args:
            position: 词汇量大小
            d_model: 深度，词嵌入维度
        Returns:
        """
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs: tf.Tensor):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def transformer_encoder_layer(units: int, d_model: int, num_heads: int,
                              dropout: float, name: str = "transformer_encoder_layer"):
    """
    Transformer的encoder层，使用函数式API
    Args:
        units: 词汇量大小
        d_model: 深度，词嵌入维度
        num_heads: 注意力头数
        dropout: dropout的权重
        name: 名称
    Returns:
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def transformer_decoder_layer(units: int, d_model: int, num_heads: int,
                              dropout: float, name: str = "transformer_decoder_layer"):
    """
    Transformer的decoder层，使用函数式API
    Args:
        units: 词汇量大小
        d_model: 深度，词嵌入维度
        num_heads: 注意力头数
        dropout: dropout的权重
        name: 名称
    Returns:
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )
