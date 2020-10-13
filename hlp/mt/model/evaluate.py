"""
对指定文本进行翻译质量评估
现有评估指标：
- BLEU指标

"""

from common import eval_bleu
import config.get_config as _config
from model import translator
from common import preprocess


# BLEU指标计算
def calc_bleu(path, transformer, tokenizer_en, tokenizer_ch):
    # 读入文本
    en, ch = preprocess.load_sentences(path, _config.num_eval)

    print('开始计算BLEU指标...')
    bleu_sum = 0
    for i in range(_config.num_eval):
        candidate_sentence = translator.translate(en[i], transformer, tokenizer_en, tokenizer_ch
                                                  , beam_size=1)[0]
        print('-' * 20)
        print('第%d个句子：' % (i + 1))
        print('英文句子:' + en[i])
        print('机翻句子:' + candidate_sentence)
        print('参考句子:' + ch[i])
        bleu_i = eval_bleu.sentence_bleu(candidate_sentence, [ch[i]], ch=True)
        print('此句子BLEU指标:%.2f' % bleu_i)
        bleu_sum += bleu_i
    bleu = bleu_sum / _config.num_eval
    print('-' * 20)
    print('平均BLEU指标为：%.2f' % bleu)


