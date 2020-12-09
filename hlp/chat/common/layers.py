import tensorflow as tf
import hlp.utils.layers as layers


def transformer_encoder_layer(units: int, d_model: int, num_heads: int,
                              dropout: float, name: str = "transformer_encoder_layer"):
    """
    Transformer的encoder层，使用函数式API
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param name: 名称
    :return: Transformer的Encoder内部层
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention, _ = layers.MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs,
                                                                 mask=padding_mask)
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
    :param units: 词汇量大小
    :param d_model: 深度，词嵌入维度
    :param num_heads: 注意力头数
    :param dropout: dropout的权重
    :param name: 名称
    :return: Transformer的Decoder内部层
    """
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1, _ = layers.MultiHeadAttention(d_model, num_heads)(q=inputs, k=inputs, v=inputs, mask=look_ahead_mask)
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2, _ = layers.MultiHeadAttention(d_model, num_heads)(q=attention1, k=enc_outputs, v=enc_outputs,
                                                                  mask=padding_mask)
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
