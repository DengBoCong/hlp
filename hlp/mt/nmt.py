import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import evaluate,trainer,translator
from optparse import OptionParser
from model import evaluate as eval
from common import preprocess
from model import network
'''

程序入口

共有三种模式：
- train: 使用 ./data 文件夹下的指定文件(默认 en-ch.txt)进行训练
- eval : 使用 ./data 文件夹下的指定文件(默认 en-ch_evaluate.txt)对模型进行评价,需对指标类型进行选择
    - bleu 指标
- translate : 对指定输入句子进行翻译，输入exit退出

cmd: python nml.py -t/--type [执行模式]

'''


def main():
    parser = OptionParser(version='%prog V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="translate",
                      help="execute type: train/bleu/translate")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        trainer.train()
    else:
        if_ckpt = preprocess.check_point()  # 检测是否有检查点
        if if_ckpt:
            network.load_checkpoint(network.transformer, network.optimizer)  # 加载检查点
            if options.type == 'eval':
                # 评估模式
                while(True):
                    print('-' * 30)
                    print('可选择评价指标： 1.bleu指标  0.退出程序')
                    eval_mode = input('请输入选择指标的序号：')
                    if eval_mode == '1':
                        eval.calc_bleu()
                    elif eval_mode == '0':
                        print('感谢您的体验！')
                        break
                    else:
                        print('请输入正确序号')
            elif options.type == 'translate':
                # 翻译模式
                while(True):
                    print('-'*30)
                    print('输入0可退出程序')
                    sentence = input('请输入要翻译的句子 :')
                    if sentence == '0':
                        break
                    else:
                        print('翻译结果:', translator.translate(sentence))
            else:
                print("请输入正确的指令")
        else:
            print('请先训练再进行测试体验...')


if __name__ == '__main__':
    main()