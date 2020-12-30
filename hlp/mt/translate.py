import hlp.mt.common.misc
from hlp.mt.model import nmt_model
from hlp.mt import translator


def main():
    if hlp.mt.common.misc.check_and_create():  # 检测是否有检查点
        # 读取保存的需要的配置
        transformer, _, tokenizer_source, tokenizer_target = nmt_model.load_model()

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
