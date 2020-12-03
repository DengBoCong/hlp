import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from common import preprocess as _pre
from common import bleu as _bleu
import config.get_config as _config
from model import translator
from common import preprocess

"""
使用保存的字典进行评估
"""


# BLEU指标计算
def _calc_bleu(path, transformer, tokenizer_source, tokenizer_target):
    # 读入文本
    source_sentences, target_sentences = preprocess.load_sentences(path, _config.num_eval)

    print('开始计算BLEU指标...')
    bleu_sum = 0
    for i in range(_config.num_eval):
        candidate_sentence = translator.translate(source_sentences[i], transformer, tokenizer_source, tokenizer_target
                                                  , beam_size=1)[0]
        print('-' * 20)
        print('第%d/%d个句子：' % (i + 1, _config.num_eval))
        print('原句子:' + source_sentences[i].strip())
        print('机翻句子:' + candidate_sentence)
        print('参考句子:' + target_sentences[i])
        bleu_i = _bleu.sentence_bleu_nltk(candidate_sentence, [target_sentences[i]], language=_config.target_lang)
        print('此句子BLEU指标:%.2f' % bleu_i)
        bleu_sum += bleu_i
    bleu = bleu_sum / _config.num_eval
    print('-' * 20)
    print('平均BLEU指标为：%.2f' % bleu)


def main():
    if _pre.check_point():  # 检测是否有检查点
        # 读取保存的需要的配置
        transformer, tokenizer_source, tokenizer_target = _pre.load_model()
        _calc_bleu(_config.path_to_eval_file, transformer, tokenizer_source, tokenizer_target)
    else:
        print('请先训练才可使用evaluate功能...')


if __name__ == '__main__':
    main()
