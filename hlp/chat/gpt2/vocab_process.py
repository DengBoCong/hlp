raw_data = './chat_data/3500_raw_vocab.txt'
output_vocab = './vocab/vocab_middle.txt'


def process(input_file, output_file):
    infopen = open(input_file, 'r', encoding="utf-8")
    outfopen = open(output_file, 'w', encoding="utf-8")
    lines = infopen.readlines()
    for data in lines:
        outfopen.write(data.replace(' ', '\n'))
    infopen.close()
    outfopen.close()


def main():
    process(raw_data, output_vocab)


if __name__ == '__main__':
    main()
