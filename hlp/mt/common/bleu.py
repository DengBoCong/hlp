import re
import nltk


def bleu_nltk(candidate_sentence, reference_sentences, language):
    """
        :param candidate_sentence:机翻句子
        :param reference_sentences:参考句子列表
        :param language:句子的语言

    """
    # 根据所选择的语言对句子进行预处理
    if language == "zh":
        candidate_sentence = [w for w in candidate_sentence]
        reference_sentences_sum = []
        for sentence in reference_sentences:
            reference_sentences_sum.append([w for w in sentence])
    elif language == "en":
        candidate_sentence = re.sub(r'([?.!,])', r' \1', candidate_sentence)  # 在?.!,前添加空格
        candidate_sentence = re.sub(r'[" "]+', " ", candidate_sentence)  # 合并连续的空格
        candidate_sentence = candidate_sentence.split(' ')
        reference_sentences_sum = []
        for sentence in reference_sentences:
            sentence = re.sub(r'([?.!,])', r' \1', sentence)  # 在?.!,前添加空格
            sentence = re.sub(r'[" "]+', " ", sentence)  # 合并连续的空格
            sentence = sentence.split(' ')
            reference_sentences_sum.append(sentence)

    smooth_function = nltk.translate.bleu_score.SmoothingFunction()
    score = nltk.translate.bleu_score.sentence_bleu(reference_sentences_sum,
                                                    candidate_sentence,
                                                    smoothing_function=smooth_function.method1)
    return score * 100


def main():
    # 测试语句
    candidate_sentence_zh = '今天的天气真好啊。'
    reference_sentence_zh = '今天可真是个好天气啊。'
    score = bleu_nltk(candidate_sentence_zh, [reference_sentence_zh], language='zh')
    print('NLTK_BLEU:%.2f' % score)

    # 测试英语语句
    candidate_sentence_en = "It's a good day."
    reference_sentence_en = "It's really a good sunny day."
    score = bleu_nltk(candidate_sentence_en, [reference_sentence_en], language='en')
    print('NLTK_BLEU:%.2f' % score)


if __name__ == '__main__':
    main()
