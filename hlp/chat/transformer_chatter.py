import tensorflow as tf
from free.chatter import Chatter
from common.common import CmdParser
import config.get_config as _config
import free.transformer.model as transformer
from common.pre_treat import preprocess_raw_data


class TransformerChatter(Chatter):
    """
    Transformer模型的聊天类
    """

    def __init__(self, checkpoint_dir, beam_size, vocab_size):
        """
        Transformer聊天器初始化，用于加载模型
        """
        super().__init__(checkpoint_dir, beam_size)

        self.model = transformer.transformer(
            vocab_size=vocab_size,
            num_layers=_config.transformer_num_layers,
            units=_config.transformer_units,
            d_model=_config.transformer_d_model,
            num_heads=_config.transformer_num_heads,
            dropout=_config.transformer_dropout
        )
        self.learning_rate = transformer.CustomSchedule(_config.transformer_d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.checkpoint = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)

        if self.ckpt:
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    def _init_loss_accuracy(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

    def _train_step(self, inp, tar, step_loss):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self.model(inputs=[inp, tar_inp])
            loss = self._loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

        step_loss[0] = self.train_loss.result()

    def _create_predictions(self, inputs, dec_input, t):
        # 获取目前已经保存在容器中的序列
        predictions = self.model(inputs=[inputs, dec_input], training=False)
        predictions = predictions[:, -1:, :]
        predictions = tf.squeeze(predictions, axis=1)
        return predictions

    def _loss_function(self, real, pred):
        real = tf.reshape(real, shape=(-1, _config.max_length_inp - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(real, pred)
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


def main():
    parser = CmdParser(version='%transformer chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    # 初始化要使用的聊天器
    chatter = TransformerChatter(checkpoint_dir=_config.transformer_train_data,
                                 beam_size=_config.beam_size,
                                 vocab_size=_config.vocab_size)

    if options.type == 'train':
        chatter.train(chatter.checkpoint,
                      dict_fn='data/transformer_dict.json',
                      data_fn=_config.data,
                      start_sign='start',
                      end_sign='end',
                      max_train_data_size=_config.max_train_data_size)
    elif options.type == 'chat':
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req, dict_fn='data/transformer_dict.json')
            print("Agent: ", response)
    elif options.type == 'pre_treat':
        preprocess_raw_data(raw_data=_config.resource_data, tokenized_data=_config.tokenized_data)
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    Transformer入口：指令需要附带运行参数
    cmd：python transformer_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()
