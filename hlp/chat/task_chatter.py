import tensorflow as tf
import task.model as task
from free.chatter import Chatter
import common.data_utils as _data
from common.common import CmdParser
import config.get_config as _config
from common.pre_treat import preprocess_raw_task_data


class TaskChatter(Chatter):
    """
    Task模型的聊天器
    """

    def __init__(self, checkpoint_dir, beam_size):
        super().__init__(checkpoint_dir, beam_size)

        self.optimizer = tf.keras.optimizers.RMSprop()
        onto, onto_idx = _data.load_ontology(_config.ontology)
        print(onto['area'])
        print(len(onto['area']))
        exit(0)
        # task.gen_tracker_model_loss
        if self.ckpt:
            print('待完善')

    def _init_loss_accuracy(self):
        print('待完善')

    def _train_step(self, inp, tar, step_loss):
        print('待完善')

    def _create_predictions(self, inputs, dec_input, t):
        print('待完善')

    def train(self, checkpoint, dict_fn, data_fn, start_sign, end_sign, max_train_data_size):
        input_tensor, target_tensor, lang_tokenizer = \
            _data.load_dataset(dict_fn=dict_fn,
                               data_fn=data_fn,
                               start_sign=start_sign,
                               end_sign=end_sign,
                               max_train_data_size=max_train_data_size)
        print('是的')


def main():
    parser = CmdParser(version='%task chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    chatter = TaskChatter(checkpoint_dir=_config.task_train_data, beam_size=_config.beam_size)

    if options.type == 'train':
        chatter.train(checkpoint='',
                      dict_fn='data/task_dict.json',
                      data_fn=_config.dialogues_tokenized,
                      start_sign='<sos>',
                      end_sign='<eos>',
                      max_train_data_size=0)
    elif options.type == 'chat':
        print('Agent: 你好！结束聊天请输入ESC。')
        while True:
            req = input('User: ')
            if req == 'ESC':
                print('Agent: 再见！')
                exit(0)
            # response = chatter.respond(req)
            response = '待完善'
            print('Agent: ', response)
    elif options.type == 'pre_treat':
        preprocess_raw_task_data(raw_data=_config.dialogues_train,
                                 tokenized_data=_config.dialogues_tokenized,
                                 semi_dict=_config.semi_dict,
                                 database=_config.database,
                                 ontology=_config.ontology)
    else:
        parser.error(msg='')


if __name__ == "__main__":
    """
    TaskModel入口：指令需要附带运行参数
    cmd：python task_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入exit即退出对话
    """
    main()
