from hlp.mt.common import bleu as _bleu
from hlp.mt.common import load_dataset
from hlp.mt.config import get_config as _config
from hlp.mt.model import nmt_model
from hlp.mt import translator
from hlp.mt.common.misc import check_and_create


# BLEU指标计算
def _calc_bleu(path, model, tokenizer_source, tokenizer_target):
    # 读入文本
    source_sentences, target_sentences = load_dataset.load_sentences(path, _config.num_eval)

    print('开始计算BLEU指标...')
    bleu_sum = 0
    for i in range(_config.num_eval):
        candidate_sentence = translator.translate(source_sentences[i], model, tokenizer_source,
                                                  tokenizer_target, beam_size=_config.BEAM_SIZE)[0]
        print('-' * 20)
        print('第%d/%d个句子：' % (i + 1, _config.num_eval))
        print('源句子:' + source_sentences[i].strip())
        print('机翻句子:' + candidate_sentence)
        print('参考句子:' + target_sentences[i])
        bleu_i = _bleu.bleu_nltk(candidate_sentence, [target_sentences[i]], language=_config.target_lang)
        print('此句子BLEU指标:%.2f' % bleu_i)
        bleu_sum += bleu_i

    bleu = bleu_sum / _config.num_eval
    print('-' * 20)
    print('平均BLEU指标为：%.2f' % bleu)


def main():
    if check_and_create(_config.checkpoint_path):  # 检测是否有检查点
        # 读取保存的需要的配置
        transformer, _, tokenizer_source, tokenizer_target = nmt_model.load_model()
        _calc_bleu(_config.path_to_eval_file, transformer, tokenizer_source, tokenizer_target)
    else:
        print('没有发现训练好的模型，请先训练模型.')


if __name__ == '__main__':
    main()
