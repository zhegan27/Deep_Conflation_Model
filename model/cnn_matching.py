
import numpy as np
import theano
import theano.tensor as tensor

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import _p
from utils import uniform_weight

from cnn_layers import param_init_encoder, encoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_chars = options['n_chars']
    img_w = options['img_w']  
    
    params = OrderedDict()
    # character embedding 
    params['Wemb'] = uniform_weight(n_chars,img_w)
    # encoding characters into words
    length = len(options['filter_shapes'])
    for idx in range(length):
        params = param_init_encoder(options['filter_shapes'][idx],params,prefix=_p('cnn_encoder',idx))
                                    
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
     
    x = tensor.matrix('x', dtype='int32')
    y = tensor.matrix('y',dtype='int32')
    cy = tensor.matrix('cy',dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1]))     
    
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input_x = tensor.concatenate(layer1_inputs,1)   
    layer1_input_x = dropout(layer1_input_x, trng, use_noise) 
    
    layer0_input = tparams['Wemb'][tensor.cast(y.flatten(),dtype='int32')].reshape((y.shape[0],1,y.shape[1],tparams['Wemb'].shape[1]))     
    
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input_y = tensor.concatenate(layer1_inputs,1)   
    layer1_input_y = dropout(layer1_input_y, trng, use_noise) 
    
    layer0_input = tparams['Wemb'][tensor.cast(cy.flatten(),dtype='int32')].reshape((cy.shape[0],1,cy.shape[1],tparams['Wemb'].shape[1]))     
    
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input_cy = tensor.concatenate(layer1_inputs,1)   
    layer1_input_cy = dropout(layer1_input_cy, trng, use_noise)
    
    layer1_input_x = l2norm(layer1_input_x)
    layer1_input_y = l2norm(layer1_input_y)
    layer1_input_cy = l2norm(layer1_input_cy)
    
    # Tile by number of contrast terms
    # (ncon*n_samples) * (2*n_h)
    layer1_input_x = tensor.tile(layer1_input_x, (options['ncon'], 1))
    layer1_input_y = tensor.tile(layer1_input_y, (options['ncon'], 1))
    
    cost = tensor.log(1+tensor.sum(tensor.exp(-options['gamma']*((layer1_input_x * layer1_input_y).sum(axis=1) - (layer1_input_x * layer1_input_cy).sum(axis=1)))))                          

    return use_noise, [x, y, cy], cost  

def build_encoder(tparams,options):
    
    x = tensor.matrix('x', dtype='int32')
    y = tensor.matrix('y',dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1]))     
    
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input_x = tensor.concatenate(layer1_inputs,1)    
    
    layer0_input = tparams['Wemb'][tensor.cast(y.flatten(),dtype='int32')].reshape((y.shape[0],1,y.shape[1],tparams['Wemb'].shape[1]))     
    
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape=filter_shape, pool_size=pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input_y = tensor.concatenate(layer1_inputs,1)   
    
    feat_x = l2norm(layer1_input_x)
    feat_y = l2norm(layer1_input_y)
    
    return [x, y], feat_x, feat_y                        
