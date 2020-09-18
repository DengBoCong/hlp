class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, num_filters,
                 kernel_size,lstm_unit, rate):
        # self.num_filters=512,kernel_size=5,rate=0.5,self.lstm_unit=256
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.lstm_unit = lstm_unit
        self.rate = rate
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        #定义嵌入层
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)
        #定义三层卷积层
        self.conv1d1 = tf.layers.conv1d(self.num_filters, self.kernel_size, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.output1 = tf.keras.layers.BatchNormalization()

        self.conv1d2 = tf.layers.conv1d(self.num_filters, self.kernel_size, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.output2 = tf.keras.layers.BatchNormalization()

        self.conv1d3 = tf.layers.conv1d(self.num_filters, self.kernel_size, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        self.output3 = tf.keras.layers.BatchNormalization()
        #定义两次LSTM
        self.forward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, return_sequences=True)
        self.backward_layer = tf.keras.layers.LSTM(units=self.lstm_unit, activation='relu', return_sequences=True,
                                            go_backwards=True)
        self.bidir = tf.keras.layers.Bidirectional()


    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.conv1d1(x)
        x = self.dropout1(x, training=training)
        x = self.output1(x)
        x = self.conv1d2(x)
        x = self.dropout2(x, training=training)
        x = self.output2(x)
        x = self.conv1d3(x)
        x = self.dropout3(x, training=training)
        x = self.output3(x)
        output = self.bidir(layer=self.forward_layer,
                                   backward_layer=self.backward_layer,retrun_sequences = True)(x)
    return output


