import tensorflow as tf


#子类化构建DS2模型
# step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
def clipped_relu(x):
    return tf.keras.activations.relu(x, max_value=20)

class DS2(tf.keras.Model):
    # dense_units=num_classes
    def __init__(
        self,
        conv_layers,
        filters,
        kernel_size,
        strides,
        bi_gru_layers,
        gru_units,
        fc_units,
        dense_units,
        **kwargs
        ):
        super(DS2,self).__init__(**kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.conv_layers = conv_layers
        self.conv = []
        for i in range(conv_layers):
            self.conv.append(tf.keras.layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    activation="relu",
                    name="conv"+str(i)
                    ))
        self.bn2 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.bi_gru_layers = bi_gru_layers
        self.bi_gru = []
        for i in range(bi_gru_layers):
            self.bi_gru.append(tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(
                            gru_units,
                            activation="relu",
                            return_sequences=True
                            ),
                    merge_mode="sum",
                    name="bi_gru"+str(i)
                    ))
        self.bn3 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(fc_units, activation=clipped_relu))
        self.sm = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_units, activation="softmax"))
    
    def call(self, inputs):
        x = inputs
        x = self.bn1(x)
        for i in range(self.conv_layers):
            x = self.conv[i](x)
        x = self.bn2(x)
        for i in range(self.bi_gru_layers):
            x = self.bi_gru[i](x)
        x = self.bn3(x)
        x = self.fc(x)
        x = self.sm(x)
        return x


# 基于模型预测得到的序列list并通过字典集来进行解码处理
def decode_output(seq, index_word, mode):
    if mode.lower() == "cn":
        return decode_output_cn_sentence(seq, index_word)
    elif mode.lower() == "en_word":
        return decode_output_en_sentence_word(seq, index_word)
    elif mode.lower() == "en_char":
        return decode_output_en_sentence_char(seq, index_word)

def decode_output_cn_sentence(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= len(index_word):
            word = index_word[str(i)]
            result += word
    return result

def decode_output_en_sentence_word(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            result += word+" "
    return result.strip()

def decode_output_en_sentence_char(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<space>":
                result += word
            else:
                result = result.strip() + " "
    return result.strip()


if __name__ == "__main__":
    pass
