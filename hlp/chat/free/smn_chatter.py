import os
import sys
import time
import model.smn as smn
import tensorflow as tf

sys.path.append(sys.path[0][:-10])
from common.utils import CmdParser
from common.utils import CustomSchedule
import common.data_utils as _data
import config.get_config as _config


class SMNChatter():
    """
    SMN的聊天器
    """

    def __init__(self, units, vocab_size, execute_type, dict_fn, embedding_dim, checkpoint_dir, max_utterance,
                 max_sentence):
        self.units = units
        self.vocab_size = vocab_size
        self.dict_fn = dict_fn
        self.checkpoint_dir = checkpoint_dir
        self.max_utterance = max_utterance
        self.max_sentence = max_sentence
        self.embedding_dim = embedding_dim
        self.learning_rate = CustomSchedule(embedding_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean()

        self.model = smn.smn(units=units, vocab_size=vocab_size,
                             embedding_dim=self.embedding_dim,
                             max_utterance=self.max_utterance,
                             max_sentence=self.max_sentence)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, )

        ckpt = os.path.exists(checkpoint_dir)
        if not ckpt:
            os.makedirs(checkpoint_dir)

        print('正在检查是否存在检查点...')
        if ckpt:
            print('存在检查点，正在从“{}”中加载检查点...'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式...')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

    def train(self, epochs, data_fn, max_train_data_size=0):
        dataset, checkpoint_prefix, steps_per_epoch = _data.smn_load_train_data(dict_fn=self.dict_fn, data_fn=data_fn,
                                                                                checkpoint_dir=self.checkpoint_dir,
                                                                                max_utterance=self.max_utterance,
                                                                                max_sentence=self.max_sentence,
                                                                                max_train_data_size=max_train_data_size)
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, _config.epochs))
            start_time = time.time()
            self.train_loss.reset_states()

            sample_sum = 0
            batch_sum = 0

            for (batch, (utterances, response, label)) in enumerate(dataset.take(steps_per_epoch)):
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs=[utterances, response])
                    # print(label)
                    # print(outputs)
                    # exit(0)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(
                        reduction=tf.keras.losses.Reduction.AUTO)(label, outputs)
                gradient = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
                self.train_loss(loss)

                sample_num = len(utterances)
                batch_sum += sample_num
                sample_sum = steps_per_epoch * sample_num
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum),
                      end='', flush=True)

            step_time = time.time() - start_time
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, self.train_loss.result()))
            sys.stdout.flush()
            self.checkpoint.save(file_prefix=checkpoint_prefix)

    def response(self, req):
        print('正在从“{}”处加载字典...'.format(self.dict_fn))
        token = _data.load_token_dict(dict_fn=self.dict_fn)
        print('功能待完善...')



def get_chatter(execute_type):
    chatter = SMNChatter(units=_config.smn_units,
                         vocab_size=_config.smn_vocab_size,
                         execute_type=execute_type,
                         dict_fn=_config.smn_dict_fn,
                         embedding_dim=_config.smn_embedding_dim,
                         checkpoint_dir=_config.smn_checkpoint,
                         max_utterance=_config.smn_max_utterance,
                         max_sentence=_config.smn_max_sentence)

    return chatter


def main():
    parser = CmdParser(version='%smn chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type)
        chatter.train(epochs=_config.epochs, data_fn=_config.ubuntu_tokenized_data, max_train_data_size=100)

    elif options.type == 'chat':
        print("待完善")
        chatter = get_chatter(execute_type=options.type)
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            response = chatter.respond(req=req)
            print("Agent: ", response)
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    SMN入口：指令需要附带运行参数
    cmd：python smn_chatter.py -t/--type [执行模式]
    执行类别：train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()
