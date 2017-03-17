
from collections import defaultdict
import cPickle

def load_data(): 
    textA, textB = [], []
    with open('./dataset.txt', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            textA.append(text[0])
            textB.append(text[1])
    
    return textA, textB
    
def build_vocab(textA, textB):
    
    vocab = defaultdict(float)
    text = textA + textB
    for seq in text:
        chars = set(list(seq))
        for char in chars:
            vocab[char] +=1
            
    ix2char = defaultdict()
    char2ix = defaultdict()
    
    count = 0
    for c in vocab.keys():
        char2ix[c] = count
        ix2char[count] = c
        count += 1
        
    return char2ix, ix2char
    
def get_idx_from_data(train,char2ix,ix2char):
    
    trainA, trainB = [], []
    
    for string in train[0]:
        seq = []
        chars = list(string)
        for c in chars:
            seq.append(char2ix[c])
        trainA.append(seq)
    
    for string in train[1]:
        seq = []
        chars = list(string)
        for c in chars:
            seq.append(char2ix[c])
        trainB.append(seq)
        
    train = [trainA,trainB]
    
    return train
        
if __name__ == '__main__':
    
    textA, textB = load_data()
    char2ix, ix2char = build_vocab(textA, textB)
    text = get_idx_from_data([textA,textB],char2ix,ix2char)
    
    cPickle.dump([text, char2ix, ix2char], open("data.p","wb"))
    