
import numpy as np
import theano
import theano.tensor as tensor

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_chars = options['n_chars']
    n_h = options['n_h']  
    
    params = OrderedDict()
    # character embedding 
    params['W1'] = uniform_weight(n_chars,n_h)
    params['b1'] = zero_bias(n_h)
    params['W2'] = uniform_weight(n_h,n_h)
    params['b2'] = zero_bias(n_h)
                                    
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        #tparams[kk].tag.test_value = params[kk]
    return tparams
    
# L2norm, row-wise
def l2norm(X):
    norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
    X /= norm[:,None]
    return X
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(SEED)
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    # n_samples * n_chars
    x = tensor.matrix('x', dtype='int32')
    y = tensor.matrix('y',dtype='int32')
    # (ncons*n_samples) * n_chars
    cy = tensor.matrix('cy',dtype='int32')
    
     # n_samples * n_h
    tmp_x = tensor.tanh(tensor.dot(x,tparams['W1'])+tparams['b1'])
    tmp_y = tensor.tanh(tensor.dot(y,tparams['W1'])+tparams['b1'])
    # (ncons*n_samples) * n_h
    tmp_cy = tensor.tanh(tensor.dot(cy,tparams['W1'])+tparams['b1'])
    
    # n_samples * n_h
    feats_x = tensor.tanh(tensor.dot(tmp_x,tparams['W2'])+tparams['b2'])
    feats_y = tensor.tanh(tensor.dot(tmp_y,tparams['W2'])+tparams['b2'])
    # (ncons*n_samples) * n_h
    feats_cy = tensor.tanh(tensor.dot(tmp_cy,tparams['W2'])+tparams['b2'])
    
    feats_x = dropout(feats_x, trng, use_noise) 
    feats_y = dropout(feats_y, trng, use_noise) 
    feats_cy = dropout(feats_cy, trng, use_noise) 
    
    feats_x = l2norm(feats_x)
    feats_y = l2norm(feats_y)
    feats_cy = l2norm(feats_cy)
    
    # Tile by number of contrast terms
    # (ncon*n_samples) * n_h
    feats_x = tensor.tile(feats_x, (options['ncon'], 1))
    feats_y = tensor.tile(feats_y, (options['ncon'], 1))
    
    cost = tensor.log(1+tensor.sum(tensor.exp(-options['gamma']*((feats_x * feats_y).sum(axis=1) - (feats_x * feats_cy).sum(axis=1)))))                          

    return use_noise, [x, y, cy], cost  

def build_encoder(tparams,options):
    
    # n_samples * n_chars
    x = tensor.matrix('x', dtype='int32')
    y = tensor.matrix('y',dtype='int32')
    
     # n_samples * n_h
    tmp_x = tensor.tanh(tensor.dot(x,tparams['W1'])+tparams['b1'])
    tmp_y = tensor.tanh(tensor.dot(y,tparams['W1'])+tparams['b1'])
    
    # n_samples * n_h
    feats_x = tensor.tanh(tensor.dot(tmp_x,tparams['W2'])+tparams['b2'])
    feats_y = tensor.tanh(tensor.dot(tmp_y,tparams['W2'])+tparams['b2'])   
    
    feat_x = l2norm(feats_x)
    feat_y = l2norm(feats_y)
    
    return [x, y], feat_x, feat_y                        
