import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import evaluate as eval
from common import preprocess as _pre
from config import get_config as _config

"""
使用保存的字典进行评估
"""


def main():
    if _pre.check_point():  # 检测是否有检查点
        # 读取保存的需要的配置
        transformer, tokenizer_source, tokenizer_target = _pre.load_model()

        # evaluate
        print('-' * 30)
        print('可选择评价指标： 1.bleu指标  0.退出程序')
        eval_mode = input('请输入选择指标的序号：')
        if eval_mode == '1':
            eval.calc_bleu(_config.path_to_eval_file, transformer, tokenizer_source, tokenizer_target)
        elif eval_mode == '0':
            print('感谢您的体验！')
        else:
            print('请输入正确序号')
    else:
        print('请先训练才可使用evaluate功能...')


if __name__ == '__main__':
    main()
