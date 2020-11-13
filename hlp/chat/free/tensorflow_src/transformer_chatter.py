import sys
import tensorflow as tf
import common.data_utils as _data
sys.path.extend([sys.path[0][:-25], sys.path[0][:-15]])
from model.chatter import Chatter
from common.utils import CmdParser
from common.utils import CustomSchedule
import config.get_config as _config
import model.transformer as transformer
from common.pre_treat import preprocess_raw_lccc_data


class TransformerChatter(Chatter):
    """
    Transformer模型的聊天类
    """

    def __init__(self, execute_type: str, checkpoint_dir: str,
                 beam_size: int, vocab_size: int, dict_fn: str, max_length: int):
        """
        Transformer聊天器初始化，用于加载模型
        Args:
            execute_type: 对话执行模式
            checkpoint_dir: 检查点保存目录路径
            beam_size: batch大小
            vocab_size: 词汇量大小
            dict_fn: 保存字典路径
            max_length: 单个句子最大长度
        Returns:
        """
        super().__init__(checkpoint_dir, beam_size, max_length)

        self.model = transformer.transformer(
            vocab_size=vocab_size,
            num_layers=_config.transformer_num_layers,
            units=_config.transformer_units,
            d_model=_config.transformer_d_model,
            num_heads=_config.transformer_num_heads,
            dropout=_config.transformer_dropout
        )

        self.learning_rate = CustomSchedule(_config.transformer_d_model)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.checkpoint = tf.train.Checkpoint(transformer=self.model, optimizer=self.optimizer)

        if execute_type == "chat":
            print('正在从“{}”处加载字典...'.format(dict_fn))
            self.token = _data.load_token_dict(dict_fn=dict_fn)
        print('正在检查是否存在检查点...')
        if self.ckpt:
            print('存在检查点，正在从“{}”中加载检查点...'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式...')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

    def _init_loss_accuracy(self):
        """
        重置损失和精度
        """
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

    def _train_step(self, inp: tf.Tensor, tar: tf.Tensor, weight: int, step_loss: list):
        """
        Args:
            inp: 输入序列
            tar: 目标序列
            weight: 样本权重序列
            step_loss: 每步损失
        Returns:
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions = self.model(inputs=[inp, tar_inp])
            loss = self._loss_function(tar_real, predictions, weight)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)

        step_loss[0] = self.train_loss.result()

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        获取目前已经保存在容器中的序列
        Args:
            inputs: 对话中的问句
            dec_input: 对话中的答句
            t: 记录时间步
        Returns:
            predictions: 预测
        """
        predictions = self.model(inputs=[inputs, dec_input], training=False)
        predictions = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions[:, -1:, :]
        predictions = tf.squeeze(predictions, axis=1)
        return predictions

    def _loss_function(self, real: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor):
        """
        用于计算预测损失，注意要将填充的0进行mask，不纳入损失计算
        Args:
            real: 真实序列
            pred: 预测序列
            weights: 样本数据的权重
        Returns:
            loss: 该batch的平均损失
        """
        real = tf.reshape(real, shape=(-1, self.max_length - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(real, pred, sample_weight=weights)
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


def get_chatter(execute_type: str):
    """
    初始化要使用的聊天器
    Args:
        execute_type: 对话执行模型
    Returns:
        chatter: 返回对应的聊天器
    """
    chatter = TransformerChatter(execute_type=execute_type,
                                 checkpoint_dir=_config.transformer_checkpoint,
                                 beam_size=_config.beam_size,
                                 vocab_size=_config.transformer_vocab_size,
                                 dict_fn=_config.transformer_dict_fn,
                                 max_length=_config.transformer_max_length)
    return chatter


def main():
    parser = CmdParser(version='%transformer chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type)
        chatter.train(chatter.checkpoint,
                      dict_fn=_config.transformer_dict_fn,
                      data_fn=_config.lccc_tokenized_data,
                      max_train_data_size=_config.transformer_max_train_data_size)
    elif options.type == 'chat':
        chatter = get_chatter(options.type)
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    elif options.type == 'pre_treat':
        preprocess_raw_lccc_data(raw_data=_config.lccc_data,
                                 tokenized_data=_config.lccc_tokenized_data)
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
