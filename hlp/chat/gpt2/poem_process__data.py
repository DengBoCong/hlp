poem_data = './poem_data/train_poem.txt'
poem_out = './poem_data/poem_raw.txt'


def process_poemdata(in_, out):
    infopen = open(in_, 'r', encoding="utf-8")
    outfopen = open(out, 'w', encoding="utf-8")
    lines = infopen.readlines()
    for data in lines:
        data = data.replace(' ', '')
        data1 = data[0:len(data) // 2 - 1]
        data2 = data1 + ' ' + data[len(data) // 2:]
        outfopen.write(data2)
    infopen.close()
    outfopen.close()


process_poemdata(poem_data, poem_out)
