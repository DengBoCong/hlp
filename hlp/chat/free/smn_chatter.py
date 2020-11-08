import os
import sys
import time
import model.smn as smn
import tensorflow as tf

sys.path.append(sys.path[0][:-10])
from common.utils import CmdParser
import common.data_utils as _data
import config.get_config as _config


class SMNChatter():
    """
    SMN的聊天器
    """

    def __init__(self, units, vocab_size, execute_type, dict_fn, embedding_dim, checkpoint_dir, max_utterance,
                 max_sentence, learning_rate, database_fn):
        self.dict_fn = dict_fn
        self.checkpoint_dir = checkpoint_dir
        self.max_utterance = max_utterance
        self.max_sentence = max_sentence
        self.database_fn = database_fn
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_loss = tf.keras.metrics.Mean()

        self.model = smn.smn(units=units, vocab_size=vocab_size,
                             embedding_dim=embedding_dim,
                             max_utterance=self.max_utterance,
                             max_sentence=self.max_sentence)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, )

        ckpt = os.path.exists(checkpoint_dir)
        if not ckpt:
            os.makedirs(checkpoint_dir)

        if execute_type == "chat":
            print('正在从“{}”处加载字典...'.format(self.dict_fn))
            self.token = _data.load_token_dict(dict_fn=self.dict_fn)
            print('正在从“{}”处加载候选回复数据库...'.format(self.database_fn))
            self.database = _data.load_token_dict(self.database_fn)
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

    def train(self, epochs, data_fn, max_train_data_size=0, max_valid_data_size=0):
        # 处理并加载训练数据，
        dataset, tokenizer, checkpoint_prefix, steps_per_epoch = \
            _data.smn_load_train_data(dict_fn=self.dict_fn,
                                      data_fn=data_fn,
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
                    scores = self.model(inputs=[utterances, response])
                    loss = tf.keras.losses. \
                        SparseCategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.AUTO)(label, scores)
                gradient = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
                self.train_loss(loss)

                sample_num = len(utterances)
                batch_sum += sample_num
                sample_sum = steps_per_epoch * sample_num
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum),
                      end='', flush=True)

            r2_1, _ = self.evaluate(valid_fn=data_fn,
                                    tokenizer=tokenizer,
                                    max_valid_data_size=max_valid_data_size)

            step_time = time.time() - start_time
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f} - R2@1：{:0.3f}\n'
                             .format(step_time, self.train_loss.result(), r2_1))
            sys.stdout.flush()
            self.checkpoint.save(file_prefix=checkpoint_prefix)

    def evaluate(self, valid_fn, dict_fn="", tokenizer=None, max_turn_utterances_num=10, max_valid_data_size=0):
        token_dict = None
        step = max_valid_data_size // max_turn_utterances_num
        if max_valid_data_size == 0:
            return None
        if dict_fn is not "":
            token_dict = _data.load_token_dict(dict_fn)
        # 处理并加载评价数据，注意，如果max_valid_data_size传
        # 入0，就直接跳过加载评价数据，也就是说只训练不评价
        valid_dataset = _data.load_smn_valid_data(data_fn=valid_fn,
                                                  max_sentence=self.max_sentence,
                                                  max_utterance=self.max_utterance,
                                                  token_dict=token_dict,
                                                  tokenizer=tokenizer,
                                                  max_turn_utterances_num=max_turn_utterances_num,
                                                  max_valid_data_size=max_valid_data_size)

        scores = tf.constant([], dtype=tf.float32)
        labels = tf.constant([], dtype=tf.int32)
        for (batch, (utterances, response, label)) in enumerate(valid_dataset.take(step)):
            score = self.model(inputs=[utterances, response])
            score = tf.nn.softmax(score, axis=-1)
            labels = tf.concat([labels, label], axis=0)
            scores = tf.concat([scores, score[:, 1]], axis=0)

        r10_1 = self._metrics_rn_1(scores, labels, num=10)
        r2_1 = self._metrics_rn_1(scores, labels, num=2)
        return r2_1, r10_1

    def respond(self, req):
        history = req[-self.max_utterance:]
        pad_sequences = [0] * self.max_sentence
        utterance = _data.dict_texts_to_sequences(history, self.token)
        utterance_len = len(utterance)

        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != self.max_utterance:
            utterance += [pad_sequences] * (self.max_utterance - utterance_len)
        utterance = tf.keras.preprocessing.sequence.pad_sequences(utterance, maxlen=self.max_sentence,
                                                                  padding="post").tolist()

        tf_idf = _data.get_tf_idf_top_k(history)
        candidates = self.database.get('-'.join(tf_idf), None)

        if candidates is None:
            return "Sorry! I didn't hear clearly, can you say it again?"
        else:
            utterances = [utterance] * len(candidates)
            responses = _data.dict_texts_to_sequences(candidates, self.token)
            responses = tf.keras.preprocessing.sequence.pad_sequences(responses, maxlen=self.max_sentence,
                                                                      padding="post")
            utterances = tf.convert_to_tensor(utterances)
            responses = tf.convert_to_tensor(responses)
            scores = self.model(inputs=[utterances, responses])
            index = tf.argmax(scores[:, 0])

            return candidates[index]

    def _metrics_rn_1(self, scores, labels, num=10):
        """
        计算Rn@k指标
        Args:
            scores: 训练所得分数
            labels: 数据标签
            num: n
        Returns:
        """
        total = 0
        correct = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                total = total + 1
                sublist = scores[i:i + num]
                if max(sublist) == scores[i]:
                    correct = correct + 1
        return float(correct) / total


def get_chatter(execute_type):
    chatter = SMNChatter(units=_config.smn_units,
                         vocab_size=_config.smn_vocab_size,
                         execute_type=execute_type,
                         dict_fn=_config.smn_dict_fn,
                         embedding_dim=_config.smn_embedding_dim,
                         checkpoint_dir=_config.smn_checkpoint,
                         max_utterance=_config.smn_max_utterance,
                         max_sentence=_config.smn_max_sentence,
                         learning_rate=_config.smn_learning_rate,
                         database_fn=_config.candidate_database)

    return chatter


def main():
    parser = CmdParser(version='%smn chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type)
        chatter.train(epochs=_config.epochs,
                      data_fn=_config.ubuntu_tokenized_data,
                      max_train_data_size=_config.smn_max_train_data_size,
                      max_valid_data_size=_config.smn_max_valid_data_size)

    elif options.type == 'pre_treat':
        _data.creat_index_dataset(data_fn=_config.ubuntu_tokenized_data,
                                  database_fn=_config.candidate_database,
                                  max_database_size=_config.smn_max_database_size)

    elif options.type == 'evaluate':
        chatter = get_chatter(execute_type=options.type)
        r2_1, r10_1 = chatter.evaluate(valid_fn=_config.ubuntu_valid_data, dict_fn=_config.smn_dict_fn,
                                       max_valid_data_size=_config.smn_max_valid_data_size)
        print("指标：R2@1-{:0.3f}，R10@1-{:0.3f}".format(r2_1, r10_1))

    elif options.type == 'chat':
        chatter = get_chatter(execute_type=options.type)
        history = []  # 用于存放历史对话
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            history.append(req)
            response = chatter.respond(req=history)
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
