"""
对指定文本进行翻译质量评估
现有评估指标：
- BLEU指标

"""

from common import bleu as _bleu
import config.get_config as _config
from model import translator
from common import preprocess


# BLEU指标计算
def calc_bleu(path, transformer, tokenizer_source, tokenizer_target):
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


