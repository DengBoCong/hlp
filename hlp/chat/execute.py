import sys
import config.getConfig as getConfig
import model.Seq2Seq.seq2seq as seq2seq

if __name__ == '__main__':
    if sys.argv[1] == 'seq2seq':
        if sys.argv[2] == 'train':
            seq2seq.train()
        elif sys.argv[2] == 'chat':
            while (True):
                sentence = input('User:')
                if sentence == 'exit':
                    break
                else:
                    print('ChatBot:', seq2seq.predict(sentence))
