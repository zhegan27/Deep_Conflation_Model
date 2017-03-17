
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import uniform_weight

from lstm_layers import param_init_encoder, encoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_chars = options['n_chars']
    n_x = options['n_x']  
    
    params = OrderedDict()
    # character embedding 
    params['Wemb'] = uniform_weight(n_chars,n_x)
    # encoding characters into words
    params = param_init_encoder(options,params,prefix='encoder_f')
    params = param_init_encoder(options,params,prefix='encoder_b')                            

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

    # description string: n_steps * n_samples
    x = tensor.matrix('x', dtype='int32')
    x_mask = tensor.matrix('x_mask', dtype=config.floatX) 
    
    y = tensor.matrix('y', dtype='int32')
    y_mask = tensor.matrix('y_mask', dtype=config.floatX)
    
    n_steps_x = x.shape[0]
    n_steps_y = y.shape[0]
    n_samples = x.shape[1]
    
    n_x = tparams['Wemb'].shape[1]
    
    # n_steps * n_samples * n_x
    x_emb = tparams['Wemb'][x.flatten()].reshape([n_steps_x,n_samples,n_x])
    y_emb = tparams['Wemb'][y.flatten()].reshape([n_steps_y,n_samples,n_x])
    
    # n_samples * n_h
    h_emb_f_x = encoder(tparams, x_emb, mask=x_mask, prefix='encoder_f')
    h_emb_b_x = encoder(tparams, x_emb[::-1], mask=x_mask[::-1], prefix='encoder_b')
    
    h_emb_f_y = encoder(tparams, y_emb, mask=y_mask, prefix='encoder_f')
    h_emb_b_y = encoder(tparams, y_emb[::-1], mask=y_mask[::-1], prefix='encoder_b')
    
    # n_samples * (2*n_h)
    h_emb_x = tensor.concatenate((h_emb_f_x,h_emb_b_x),axis=1) 
    h_emb_y = tensor.concatenate((h_emb_f_y,h_emb_b_y),axis=1)                                                                                 
    h_emb_x = dropout(h_emb_x, trng, use_noise) 
    h_emb_y = dropout(h_emb_y, trng, use_noise) 
    
    h_emb_x = l2norm(h_emb_x)
    h_emb_y = l2norm(h_emb_y)
    
    # contrastive strings 
    # description string: n_steps * (ncon*n_samples)
    cy = tensor.matrix('cy', dtype='int32')
    cy_mask = tensor.matrix('cy_mask', dtype=config.floatX)
    
    n_steps_cy = cy.shape[0]
    n_samples_c = cy.shape[1]
    
    # n_steps * (ncon*n_samples) * n_x
    cy_emb = tparams['Wemb'][cy.flatten()].reshape([n_steps_cy,n_samples_c,n_x])
    
    # (ncon*n_samples) * n_h
    h_emb_f_cy = encoder(tparams, cy_emb, mask=cy_mask, prefix='encoder_f')
    h_emb_b_cy = encoder(tparams, cy_emb[::-1], mask=cy_mask[::-1], prefix='encoder_b')
    
    # (ncon*n_samples) * (2*n_h)
    h_emb_cy = tensor.concatenate((h_emb_f_cy,h_emb_b_cy),axis=1)                                                                                 
    h_emb_cy = dropout(h_emb_cy, trng, use_noise) 
    
    h_emb_cy = l2norm(h_emb_cy)    
    
    # Tile by number of contrast terms
    # (ncon*n_samples) * (2*n_h)
    h_emb_x = tensor.tile(h_emb_x, (options['ncon'], 1))
    h_emb_y = tensor.tile(h_emb_y, (options['ncon'], 1))
    
    cost = tensor.log(1+tensor.sum(tensor.exp(-options['gamma']*((h_emb_x * h_emb_y).sum(axis=1) - (h_emb_x * h_emb_cy).sum(axis=1)))))                          

    return use_noise, [x, x_mask, y, y_mask, cy, cy_mask], cost
    
def build_encoder(tparams,options):
    
    # description string: n_steps * n_samples
    x = tensor.matrix('x', dtype='int32')
    x_mask = tensor.matrix('x_mask', dtype=config.floatX) 
    
    y = tensor.matrix('y', dtype='int32')
    y_mask = tensor.matrix('y_mask', dtype=config.floatX)
    
    n_steps_x = x.shape[0]
    n_steps_y = y.shape[0]
    n_samples = x.shape[1]
    
    n_x = tparams['Wemb'].shape[1]
    
    # n_steps * n_samples * n_x
    x_emb = tparams['Wemb'][x.flatten()].reshape([n_steps_x,n_samples,n_x])
    y_emb = tparams['Wemb'][y.flatten()].reshape([n_steps_y,n_samples,n_x])
    
    # n_samples * n_h
    h_emb_f_x = encoder(tparams, x_emb, mask=x_mask, prefix='encoder_f')
    h_emb_b_x = encoder(tparams, x_emb[::-1], mask=x_mask[::-1], prefix='encoder_b')
    
    h_emb_f_y = encoder(tparams, y_emb, mask=y_mask, prefix='encoder_f')
    h_emb_b_y = encoder(tparams, y_emb[::-1], mask=y_mask[::-1], prefix='encoder_b')
    
    # n_samples * (2*n_h)
    h_emb_x = tensor.concatenate((h_emb_f_x,h_emb_b_x),axis=1) 
    h_emb_y = tensor.concatenate((h_emb_f_y,h_emb_b_y),axis=1)                                                                                 
    
    feat_x = l2norm(h_emb_x)
    feat_y = l2norm(h_emb_y)
    
    return [x, x_mask, y, y_mask], feat_x, feat_y
