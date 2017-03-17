
from collections import defaultdict
import cPickle

def load_data(): 
    trainA, trainB = [], []
    with open('./train.pair', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            trainA.append(text[0])
            trainB.append(text[1])
            
    validA, validB = [], []
    with open('./validtest.pair', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            validA.append(text[0])
            validB.append(text[1])
    
    testA, testB = [], []
    with open('./test.pair', 'rb') as f:
        for line in f:
            text = line.strip().split('\t')
            testA.append(text[0])
            testB.append(text[1])
            
    train = [trainA,trainB]
    valid = [validA,validB]
    test = [testA,testB]
    
    return train, valid, test
    
def build_vocab(train, valid, test):
    
    vocab = defaultdict(float)
    text = train[0] + train[1] + valid[0] + valid[1] + test[0] + test[1]
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
    
    train, valid, test = load_data()
    char2ix, ix2char = build_vocab(train, valid, test)
    train = get_idx_from_data(train,char2ix,ix2char)
    valid = get_idx_from_data(valid,char2ix,ix2char)
    test = get_idx_from_data(test,char2ix,ix2char)
    
    cPickle.dump([train, valid, test, char2ix, ix2char], open("data_predefined_split.p","wb"))
    