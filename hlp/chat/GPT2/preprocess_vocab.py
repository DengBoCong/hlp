file_path='vocab_500.txt'
import io
import jieba
input_file='vocab/vocab_500.txt'
output_file='vocab/vocab.txt'
def cutdata(input_file,output_file):

    data=open(input_file,'r',encoding="utf-8")
    out=open(output_file,'w',encoding="utf-8")
    print(data)
    db=data.read()
    #将语料进行切割
        #print(word)
    out.write(db.replace('、','\n'))
    out.write(db.replace(' ', '\n'))
    data.close()
    out.close()

cutdata(input_file,output_file)
