import tensorflow as tf


#子类化构建DS2模型
# step1：1-3 Conv1D -> 1BN -> 1-3 bi_gru -> 1BN -> 1dense
def clipped_relu(x):
    return tf.keras.activations.relu(x, max_value=20)
class DS2(tf.keras.Model):
    #dense_units=num_classes
    def __init__(
        self,
        filters,
        kernel_size,
        strides,
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
        self.conv1 = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                activation="relu"
                )
        self.conv2 = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                activation="relu"
                )
        self.conv3 = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="valid",
                activation="relu"
                )
        self.bn2 = tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001
                )
        self.bi_gru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                        gru_units,
                        activation="relu",
                        return_sequences=True
                        ),
                merge_mode="sum"
                )
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.bi_gru(x)
        x = self.bn3(x)
        x = self.fc(x)
        x = self.sm(x)
        return x

# 基于模型预测得到的序列list并通过字典集来进行解码处理
def decode_output(seq, index_word, mode):
    if mode.lower() == "cn":
        return decode_output_ch_sentence(seq, index_word)
    elif mode.lower() == "en_word":
        return decode_output_en_sentence_word(seq, index_word)
    elif mode.lower() == "en_char":
        return decode_output_en_sentence_char(seq, index_word)

def decode_output_ch_sentence(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= len(index_word):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word
                else:
                    return result
    return result

def decode_output_en_sentence_word(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word+" "
                else:
                    return result
    return result

def decode_output_en_sentence_char(seq, index_word):
    result = ""
    for i in seq:
        if i >= 1 and i <= (len(index_word)):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    if word !="<space>":
                        result += word
                    else:
                        word += " "
                else:
                    return result
    return result


if __name__ == "__main__":
    pass
