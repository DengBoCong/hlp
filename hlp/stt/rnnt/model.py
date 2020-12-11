import tensorflow as tf


# 时间缩减层
class TimeReduction(tf.keras.layers.Layer):
    def __init__(self,
                 reduction_factor,
                 **kwargs):
        super(TimeReduction, self).__init__(**kwargs)

        self.reduction_factor = reduction_factor

    def call(self, inputs):
        batch_size = inputs.shape[0]

        max_time = inputs.shape[1]
        num_units = inputs.shape[-1]

        paddings = [[0, 0], [0, tf.floormod(max_time, self.reduction_factor)], [0, 0]]
        outputs = tf.pad(inputs, paddings)

        return tf.reshape(outputs, (batch_size, -1, num_units * self.reduction_factor))


# 编码器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, configs, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001
        )

        self.encoder_layers = configs["encoder_layers"]
        self.lstm = []
        self.dense = []
        self.dropout = []
        self.ln = []
        for i in range(self.encoder_layers):
            self.lstm.append(
                tf.keras.layers.LSTM(
                    configs["encoder_lstm_units"],
                    return_sequences=True
                )
            )
            self.dense.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(configs["proj_size"])
                )
            )
            self.dropout.append(
                tf.keras.layers.Dropout(configs["dropout"])
            )
            self.ln.append(
                tf.keras.layers.LayerNormalization()
            )
        self.reduction_factor = configs["reduction_factor"]
        self.tr = TimeReduction(self.reduction_factor)

    def call(self, inputs):
        x = self.bn(inputs)
        for i in range(self.encoder_layers):
            x = self.lstm[i](x)
            x = self.dense[i](x)
            x = self.dropout[i](x)
            x = self.ln[i](x)

            if i == self.reduction_factor:
                x = self.tr(x)

        return x


# 预测网络
class PredictionNetwork(tf.keras.layers.Layer):
    def __init__(self, configs, **kwargs):
        super(PredictionNetwork, self).__init__(**kwargs)

        self.embedding_layer = tf.keras.layers.Embedding(
            configs["vocab_size"],
            configs["embedding_size"]
        )

        self.prediction_network_layers = configs["prediction_network_layers"]
        self.lstm = []
        self.dense = []
        self.dropout = []
        self.ln = []
        for i in range(self.prediction_network_layers):
            self.lstm.append(
                tf.keras.layers.LSTM(
                    configs["prediction_network_lstm_units"],
                    return_sequences=True
                )
            )
            self.dense.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(configs["proj_size"])
                )
            )
            self.dropout.append(
                tf.keras.layers.Dropout(configs["dropout"])
            )
            self.ln.append(
                tf.keras.layers.LayerNormalization()
            )

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        for i in range(self.prediction_network_layers):
            x = self.lstm[i](x)
            x = self.dense[i](x)
            x = self.dropout[i](x)
            x = self.ln[i](x)

        return x


# RNNT，将Encoder和预测网络拼接
class RNNT(tf.keras.Model):
    def __init__(self, configs, **kwargs):
        super(RNNT, self).__init__(**kwargs)

        self.encoder = Encoder(configs)
        self.prediction_network = PredictionNetwork(configs)
        self.ds1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                configs["joint_dense_units"],
                activation="tanh"
            )
        )
        self.ds2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                configs["vocab_size"]
            )
        )

    def call(self, encoder_inputs, pre_inputs):
        encoder_outputs = self.encoder(encoder_inputs)
        pred_outputs = self.prediction_network(pre_inputs)

        # 拼接(joint):[B, T, U, V]
        joint_inputs = (
            # [B, T, V] => [B, T, 1, V]
                tf.expand_dims(encoder_outputs, axis=2) +
                # [B, U, V] => [B, 1, U, V]
                tf.expand_dims(pred_outputs, axis=1)
        )

        joint_outputs = self.ds1(joint_inputs)
        outputs = self.ds2(joint_outputs)

        return outputs


if __name__ == "__main__":
    pass
