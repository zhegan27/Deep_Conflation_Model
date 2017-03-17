'''
Training code for implementing Deep Conflation Model using Bag-of-Characters encoder
'''

import sys
import logging
import cPickle

from sklearn.cross_validation import KFold

import numpy as np
import theano
import theano.tensor as tensor

from model.boc_matching import init_params, init_tparams, build_model, build_encoder
from model.optimizers import Adam
from model.utils import unzip, zipp

from numpy.random import RandomState

def prepare_data(seqs_x, n_chars):
    
    npts = len(seqs_x)
    x = np.zeros((npts,n_chars)).astype('int32')
    
    for i in range(npts):    
        rev = seqs_x[i]
        for idx in rev:
            x[i,idx] = x[i,idx] + 1
            
    return x

# trainer
def trainer(train, valid, test, n_chars=32, n_h=300, max_epochs=20, gamma=10, 
            ncon=50, lrate=0.0002, batch_size=100, dispFreq=10, validFreq=100, 
            saveto='example.npz'):

    """ train, valid, test : datasets
        n_chars : vocabulary size
        img_w : character embedding dimension.
        max_len : the maximum length of a sentence 
        feature_maps : the number of feature maps we used 
        filter_hs: the filter window sizes we used
        max_epochs : The maximum number of epoch to run
        gamma: hyper-parameter using in ranking
        ncon: the number of negative samples we used for each postive sample
        lrate : learning rate
        batch_size : batch size during training
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation rank score after this number of update.
        saveto: where to save the result.
    """
    
    model_options = {}
    model_options['n_chars'] = n_chars
    model_options['n_h'] = n_h
    model_options['max_epochs'] = max_epochs
    model_options['gamma'] = gamma
    model_options['ncon'] = ncon
    model_options['lrate'] = lrate
    model_options['batch_size'] = batch_size
    model_options['dispFreq'] = dispFreq
    model_options['validFreq'] = validFreq
    model_options['saveto'] = saveto

    logger.info('Model options {}'.format(model_options))

    logger.info('Building model...')
    
    params = init_params(model_options)
    tparams = init_tparams(params)

    use_noise, inps, cost = build_model(tparams, model_options)

    logger.info('Building encoder...')
    inps_e, feat_x, feat_y = build_encoder(tparams, model_options)
    
    logger.info('Building functions...')
    f_emb = theano.function(inps_e, [feat_x, feat_y],name='f_emb')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, inps, lr)
    
    logger.info('Training model...')

    uidx = 0
    seed = 1234
    curr = 0
    history_errs = []
    
    valid_x = prepare_data(valid[0],n_chars)
    valid_y = prepare_data(valid[1],n_chars)
    
    test_x = prepare_data(test[0],n_chars)
    test_y = prepare_data(test[1],n_chars)

    # Main loop
    for eidx in range(max_epochs):
        prng = RandomState(seed - eidx - 1)
        
        trainA = train[0]
        trainB = train[1]
        
        num_samples = len(trainA)
        
        inds = np.arange(num_samples)
        prng.shuffle(inds)
        numbatches = len(inds) / batch_size
        for minibatch in range(numbatches):
            use_noise.set_value(0.)
            uidx += 1
            conprng = RandomState(seed + uidx + 1)

            x = [trainA[seq] for seq in inds[minibatch::numbatches]]
            y = [trainB[seq] for seq in inds[minibatch::numbatches]]

            cinds = conprng.random_integers(low=0, high=num_samples-1, size=ncon * len(x))
            cy = [trainB[seq] for seq in cinds]
            
            x = prepare_data(x,n_chars)
            y = prepare_data(y,n_chars)
            cy = prepare_data(cy,n_chars)
            
            cost = f_grad_shared(x,y,cy)
            f_update(lrate)

            if np.mod(uidx, dispFreq) == 0:
                logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))

            if np.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                logger.info('Computing ranks...')
                
                feats_x, feats_y = f_emb(valid_x, valid_y)
                (r1, r3, r10, medr, meanr,h_meanr) = rank(feats_x, feats_y)
                history_errs.append([r1, r3, r10, medr, meanr,h_meanr])
                
                logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr,meanr,h_meanr))

                currscore = r1 + r3 + r10 
                if currscore > curr:
                    curr = currscore
                    logger.info('Saving...')
                    params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    logger.info('Done...')
    
    use_noise.set_value(0.)
    zipp(params,tparams)
    logger.info('Final results...')
    
    feats_x, feats_y = f_emb(valid_x, valid_y)
    (r1, r3, r10, medr, meanr,h_meanr) = rank(feats_x, feats_y)
    logger.info('Valid Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr,meanr,h_meanr))
    
    feats_x, feats_y = f_emb(test_x, test_y)
    (r1, r3, r10, medr, meanr,h_meanr) = rank(feats_x, feats_y)
    logger.info('Test Rank:{}, {}, {}, {},{},{}'.format(r1, r3, r10, medr,meanr,h_meanr))
    
    # np.savez("./boc_feats.npz", feats_x=feats_x, feats_y=feats_y)
    
    return (r1, r3, r10, medr, meanr, h_meanr)

def rank(x, y):
    """ x,y: (n_samples, n_feats) 
    """
    npts = x.shape[0]
    n_feats = x.shape[1]
    
    index_list = []

    ranks = np.zeros(npts)
    for index in range(npts):

        # Get query text
        im = x[index].reshape(1,n_feats)

        # Compute scores
        d = np.dot(im, y.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        ranks[index] =  np.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = np.mean(ranks) + 1
    h_meanr = 1./np.mean(1./(ranks+1))
    return (r1, r3, r10, medr, meanr,h_meanr)
    
def create_valid(train_set,valid_portion=0.10):
    
    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)

    return train, valid

if __name__=="__main__":  
    
    # using predefined split or doing 10-fold cross validation
    data = sys.argv[1]
    # using the correct name to query the wrong name or reverse
    query = sys.argv[2] 
    
    # https://docs.python.org/2/howto/logging-cookbook.html
    logger = logging.getLogger('eval_boc_{}_{}'.format(data, query))
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_boc_{}_{}.log'.format(data, query))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    saveto = "boc_results_{}_{}".format(data, query)
    
    if data == "predefined_split":
    
        x = cPickle.load(open("./data/data_predefined_split.p","rb"))
        train, valid, test, char2ix, ix2char = x[0], x[1], x[2], x[3], x[4]
        del x
        
        if query == "reverse":
            trainA, trainB = train[0], train[1]
            validA, validB = valid[0], valid[1]
            testA, testB = test[0], test[1]
            
            train = [trainB,trainA]
            valid = [validB,validA]
            test = [testB, testA]
        
        (r1, r3, r10, medr, meanr, h_meanr) = trainer(train, valid, test, 
                n_chars=len(ix2char),saveto=saveto)
    
    elif data == "cross_validation":
        
        x = cPickle.load(open("./data/data.p","rb"))
        text, char2ix, ix2char = x[0], x[1], x[2]
        del x
        
        if query == "normal":
            textA = text[0]
            textB = text[1]
        elif query == "reverse":
            textA = text[1]
            textB = text[0]
    
        results = []
        i = 0
        kf = KFold(len(textA), n_folds=10, random_state=1234)
        for train_index, test_index in kf:
            train_index = train_index.tolist()
            test_index = test_index.tolist()
    
            train_x = [textA[ix] for ix in train_index]
            train_y = [textB[ix] for ix in train_index]
    
            test_x = [textA[ix] for ix in test_index]
            test_y = [textB[ix] for ix in test_index]
            
            train = (train_x, train_y)
            test = (test_x, test_y)
            train, valid = create_valid(train, valid_portion=0.10)
            (r1, r3, r10, medr, meanr, h_meanr) = trainer(train, valid, test, 
                    n_chars=len(ix2char),saveto=saveto)
            
            logger.info('cv: {} test rank: {}, {}, {}, {},{},{}'.format(i, r1, r3, r10, medr, meanr, h_meanr))
            i = i + 1
            results.append([r1, r3, r10, medr, meanr, h_meanr])
        
        np.savez("./boc_ten_fold_{}_{}.npz".format(data, query), results=results)


