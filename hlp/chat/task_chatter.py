import sys
from chit.chatter import Chatter
from common.common import CmdParser
import config.get_config as _config
from common.pre_treat import preprocess_raw_task_data


class TaskChatter(Chatter):
    """
    Task模型的聊天器
    """

    def __init__(self, checkpoint_dir, beam_size):
        super().__init__(checkpoint_dir, beam_size)
        if self.ckpt:
            print('待完善')

    def init_loss_accuracy(self):
        print('待完善')

    def train_step(self, inp, tar, step_loss):
        print('待完善')

    def create_predictions(self, inputs, dec_input, t):
        print('待完善')


def main():
    parser = CmdParser(version='%task chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()

    chatter = TaskChatter(checkpoint_dir=_config.task_train_data, beam_size=_config.beam_size)

    if options.type == 'train':
        print('待完善')
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
