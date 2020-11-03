import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import evaluate, trainer, translator
from common import preprocess as _pre


def main():
    if _pre.check_point():  # 检测是否有检查点
        # 读取保存的需要的配置
        transformer, tokenizer_source, tokenizer_target = _pre.load_model()

        # translate
        while True:
            print('-' * 30)
            print('输入0可退出程序')
            sentence = input('请输入要翻译的句子 :')
            if sentence == '0':
                break
            else:
                print('翻译结果:', translator.translate(sentence, transformer, tokenizer_source, tokenizer_target))
    else:
        print('请先训练才可使用翻译功能...')


if __name__ == '__main__':
    main()
