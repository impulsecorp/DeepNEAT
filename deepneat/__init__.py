import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
import sys
sys.path.append('/home/peter')
sys.path.append('/home/ubuntu')
#from universal import *
sys.path.append('/home/peter/code/projects')
sys.path.append('/home/ubuntu')
sys.path.append('/home/ubuntu/new/automl')
from aidevutil import *
from keras.utils.vis_utils import plot_model
from tqdm import tqdm_notebook as tqdm
import time
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
import graphviz
from collections import OrderedDict

from dask import compute, delayed, persist
from dask.distributed import Client, wait
from dask.distributed import as_completed






# NEAT Parameters
params = NEAT.Parameters()

params.PopulationSize = 100
params.DynamicCompatibility = True
params.YoungAgeTreshold = 2
params.SpeciesMaxStagnation = 10
params.OldAgeTreshold = 10
params.MinSpecies = 4
params.MaxSpecies = 8
params.RouletteWheelSelection = False
params.ArchiveEnforcement = False

params.MutateAddNeuronProb = 0.2
params.MutateAddLinkProb = 0.1
params.MutateRemLinkProb = 0.0
params.RecurrentProb = 0.0

params.MutateWeightsProb = 0.0
params.MutateActivationAProb = 0.0
params.MutateActivationBProb = 0.0
params.MutateNeuronTimeConstantsProb = 0.0
params.MutateNeuronBiasesProb = 0.0

params.MutateGenomeTraitsProb = 0.4
params.MutateNeuronTraitsProb = 0.8
params.MutateLinkTraitsProb = 0.0

params.OverallMutationRate = 0.75
params.CrossoverRate = 0.75
params.MultipointCrossoverRate = 0.5
params.SurvivalRate = 0.25

params.DontUseBiasNeuron = True
params.AllowLoops = False
params.AllowClones = False

params.ExcessCoeff = 1.5
params.DisjointCoeff = 1.0

params.WeightDiffCoeff = 0
params.TimeConstantDiffCoeff = 0
params.BiasDiffCoeff = 0
params.ActivationADiffCoeff = 0
params.ActivationBDiffCoeff = 0

params.MinCompatTreshold = 0.02
params.CompatTreshold = 1.25
params.CompatTreshChangeInterval_Evaluations = 1
params.CompatTresholdModifier = 0.1

# Traits setup

layer_set = ['dense', 'dropout', 'actreg', 'noise', 'dropnoise', 'adropout', 'gdropout',
         'conv1d', 'conv2d', 'conv2dt', 'lc1d', 'lc2d',  # 'spconv2d' not implemented yet
         'maxpool1d', 'maxpool2d', 'avgpool1d', 'avgpool2d',
         'gmaxpool1d', 'gmaxpool2d', 'gavgpool1d', 'gavgpool2d',
         'lstm', 'gru', 'srnn']
activation_set = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh',
                  'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid', ]

knownname_set = ['bnorm', 'leakyrelu', 'prelu', 'trelu']+\
                ['mm_add', 'mm_sub', 'mm_mul', 'mm_avg', 'mm_max', 'mm_min', 'mm_cnc']

def init_traits(params,
                cont_space=True,
                domain_type='all',
                activations='all',
                optimizers='all',
                losses='all',
                mode='regression'):
    domain_type, activations, optimizers, losses, mode = [x.lower() for x in
                                                          [domain_type, activations, optimizers, losses, mode]]

    # Core layer types and layer parameters
    s = ['dense', 'dropout', 'actreg', 'noise', 'dropnoise', 'adropout', 'gdropout',
         'conv1d', 'conv2d', 'conv2dt', 'lc1d', 'lc2d',  # 'spconv2d' not implemented yet
         'maxpool1d', 'maxpool2d', 'avgpool1d', 'avgpool2d',
         'gmaxpool1d', 'gmaxpool2d', 'gavgpool1d', 'gavgpool2d',
         'lstm', 'gru', 'srnn', 'bnorm']

    sa = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh',
          'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid',
          'leakyrelu', 'prelu', 'trelu']

    actlayerprob = 0.08

    p = [1.0] * len(s)
    p += [actlayerprob]*len(sa)

    if domain_type == 'visual':
        # remove everything non 2D CNN-related
        s = ['dense', 'dropout', 'conv2d', 'conv2dt', 'lc2d', 'maxpool2d', 'avgpool2d', 'bnorm']
        sa = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh', 'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid',
          'leakyrelu', 'prelu', 'trelu']
        p = [0.2, 0.5, 1.0, 0.1, 0.05, 1.0, 0.75, 0.75] + [actlayerprob]*len(sa)

    if domain_type == 'temporal':
        # remove everything non temporal related
        s = ['dense', 'dropout', 'actreg', 'noise', 'dropnoise',
             'conv1d', 'lc1d',
             'maxpool1d', 'avgpool1d',
             'gmaxpool1d', 'gavgpool1d',
             'lstm', 'gru', 'srnn', 'bnorm']
        sa = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh', 'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid',
          'leakyrelu', 'prelu', 'trelu']
        p = [1.0] * len(s)
        p += [actlayerprob] * len(sa)

    if domain_type == 'simple':
        s = ['dense', 'dropout', 'bnorm', 'actreg']
        sa = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh', 'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid',
          'leakyrelu', 'prelu', 'trelu']
        p = [1.0, 0.5, 0.5, 0.1] + [actlayerprob]*len(sa)
    if domain_type == '1d':
        s = ['dense', 'dropout', 'bnorm', 'actreg', 'maxpool1d', 'avgpool1d', 'gmaxpool1d', 'gavgpool1d', 'conv1d',
             'lc1d', 'noise', 'dropnoise', ]
        sa = ['relu', 'elu', 'selu', 'softmax', 'linear', 'tanh', 'softsign', 'softplus', 'exp', 'hsigmoid', 'sigmoid',
          'leakyrelu', 'prelu', 'trelu']
        p = [1.0, 0.5, 0.5, 0.1, 0.1, 0.05, 0.02, 0.02, 0.05, 0.02, 0.05, 0.05] + [actlayerprob]*len(sa)

    lt = {'details': {'set': s+sa, 'probs': p},
          'importance_coeff': 1.0,
          'mutation_prob': 0.2,
          'type': 'str'}

    s = ['mm_add', 'mm_sub', 'mm_mul', 'mm_avg', 'mm_max', 'mm_min', 'mm_cnc']
    p = [1.0] * len(s)
    mm = {'details': {'set': s, 'probs': p},
          'importance_coeff': 0.0,
          'mutation_prob': 0.02,
          'type': 'str'}

    if not cont_space:
        s = [8, 16, 32, 64, 128, 256]
        p = [1.0] * len(s)
        size = {'details': {'set': s, 'probs': p},
                'importance_coeff': 0.0,
                'mutation_prob': 0.1,
                'type': 'intset',

                'dep_key': 'lt',  # depends on layer type being dense
                'dep_values': ['dense', 'lstm', 'gru', 'srnn']}
    else:
        size = {'details': {'min': 8, 'max': 1024, 'mut_power': 128, 'mut_replace_prob': 0.15},
                'importance_coeff': 0.0,
                'mutation_prob': 0.1,
                'type': 'int',

                'dep_key': 'lt',  # depends on layer type being dense
                'dep_values': ['dense', 'lstm', 'gru', 'srnn']}

    # activation type
    #at = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'linear']
    #if activations == 'reduced':
    #    at = ['linear', 'relu', 'tanh', ]
    #if activations == 'minimal':
    #    at = ['relu']
    #act = {'details': {'set': at, 'probs': [1.0] * len(at)},
    #       'importance_coeff': 0.5,
    #       'mutation_prob': 0.5,
    #       'type': 'str',
    #
    #       'dep_key': 'lt',  # depends on layer type being
    #       'dep_values': ['dense', 'conv1d', 'conv2d', 'spconv2d',
    #                      'conv2dt', 'lc1d', 'lc2d', 'lstm', 'gru', 'srnn']}

    # dropout/noise parameter
    if not cont_space:
        s = [0.1, 0.25, 0.5, 0.75, 0.9]
        p = [1.0] * len(s)
        dr = {'details': {'set': s, 'probs': p},
              'importance_coeff': 0.0,
              'mutation_prob': 0.1,
              'type': 'floatset',

              'dep_key': 'lt',  # depends on layer type being dropout
              'dep_values': ['dropout', 'dropnoise', 'noise']}
    else:
        dr = {'details': {'min': 0.0, 'max': 1.0, 'mut_power': 0.2, 'mut_replace_prob': 0.25},
              'importance_coeff': 0.0,
              'mutation_prob': 0.1,
              'type': 'float',

              'dep_key': 'lt',  # depends on layer type being dropout
              'dep_values': ['dropout', 'dropnoise', 'noise']}

    # activity regularization parameters
    if not cont_space:
        s = [0.1, 0.25, 0.5, 0.75, 0.9]
        p = [1.0] * len(s)
        ar1 = {'details': {'set': s, 'probs': p},
               'importance_coeff': 0.0,
               'mutation_prob': 0.1,
               'type': 'floatset',

               'dep_key': 'lt',  # depends on layer type being activity regularization
               'dep_values': ['actreg']}
        s = [0.1, 0.25, 0.5, 0.75, 0.9]
        p = [1.0] * len(s)
        ar2 = {'details': {'set': s, 'probs': p},
               'importance_coeff': 0.0,
               'mutation_prob': 0.1,
               'type': 'floatset',

               'dep_key': 'lt',  # depends on layer type being activity regularization
               'dep_values': ['actreg']}
    else:
        ar1 = {'details': {'min': 0.1, 'max': 0.9, 'mut_power': 0.2, 'mut_replace_prob': 0.25},
               'importance_coeff': 0.0,
               'mutation_prob': 0.1,
               'type': 'float',

               'dep_key': 'lt',  # depends on layer type being activity regularization
               'dep_values': ['actreg']}

        ar2 = {'details': {'min': 0.1, 'max': 0.9, 'mut_power': 0.2, 'mut_replace_prob': 0.25},
               'importance_coeff': 0.0,
               'mutation_prob': 0.1,
               'type': 'float',

               'dep_key': 'lt',  # depends on layer type being activity regularization
               'dep_values': ['actreg']}

    params.SetNeuronTraitParameters('lt', lt)
    params.SetNeuronTraitParameters('mm', mm)
    params.SetNeuronTraitParameters('size', size)
    #params.SetNeuronTraitParameters('act', act)

    params.SetNeuronTraitParameters('dr', dr)

    params.SetNeuronTraitParameters('ar1', ar1)
    params.SetNeuronTraitParameters('ar2', ar2)

    # Initializers

    # Convolutional layers
    if not cont_space:
        s = [8, 16, 32, 64, 128]
        p = [1.0] * len(s)
        filters = {'details': {'set': s, 'probs': p},
                   'importance_coeff': 0.0,
                   'mutation_prob': 0.2,
                   'type': 'intset',

                   'dep_key': 'lt',  # depends on layer type being conv1d
                   'dep_values': ['conv1d', 'conv2d', 'spconv2d', 'conv2dt', 'lc1d', 'lc2d']}
    else:
        filters = {'details': {'min': 2, 'max': 128, 'mut_power': 16, 'mut_replace_prob': 0.25},
                   'importance_coeff': 0.0,
                   'mutation_prob': 0.2,
                   'type': 'int',

                   'dep_key': 'lt',  # depends on layer type being conv1d
                   'dep_values': ['conv1d', 'conv2d', 'spconv2d', 'conv2dt', 'lc1d', 'lc2d']}

    # Kernel size
    if not cont_space:
        s = [2, 3, 4]
        p = [1.0] * len(s)
        ksize = {'details': {'set': s, 'probs': p},
                 'importance_coeff': 0.0,
                 'mutation_prob': 0.15,
                 'type': 'intset',

                 'dep_key': 'lt',  # depends on layer type being conv1d
                 'dep_values': ['conv1d', 'conv2d', 'spconv2d', 'conv2dt', 'lc1d', 'lc2d']}
    else:
        ksize = {'details': {'min': 1, 'max': 7, 'mut_power': 2, 'mut_replace_prob': 0.25},
                 'importance_coeff': 0.0,
                 'mutation_prob': 0.15,
                 'type': 'int',

                 'dep_key': 'lt',  # depends on layer type being conv1d
                 'dep_values': ['conv1d', 'conv2d', 'spconv2d', 'conv2dt', 'lc1d', 'lc2d']}

    # Padding
    pdd = ['valid', 'same']
    padding = {'details': {'set': pdd, 'probs': [1.0] * len(pdd)},
               'importance_coeff': 0.0,
               'mutation_prob': 0.05,
               'type': 'str',

               'dep_key': 'lt',  # depends on layer type being conv1d
               'dep_values': ['conv1d', 'conv2d', 'spconv2d', 'conv2dt', 'lc1d', 'lc2d']}

    params.SetNeuronTraitParameters('filters', filters)
    params.SetNeuronTraitParameters('ksize', ksize)
    params.SetNeuronTraitParameters('padding', padding)

    # Pooling layers
    s = [2, 3, 4]
    p = [1.0] * len(s)
    psize = {'details': {'set': s, 'probs': p},
             'importance_coeff': 0.0,
             'mutation_prob': 0.1,
             'type': 'intset',

             'dep_key': 'lt',  # depends on layer type being
             'dep_values': ['maxpool1d', 'maxpool2d', 'avgpool1d', 'avgpool2d']}

    params.SetNeuronTraitParameters('psize', psize)

    # LSTM layer parameters
    at = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'linear']
    ratt = {'details': {'set': at, 'probs': [1.0] * len(at)},
            'importance_coeff': 0.0,
            'mutation_prob': 0.15,
            'type': 'str',

            'dep_key': 'lt',  # depends on layer type being lstm
            'dep_values': ['lstm', 'gru']}

    seq = {'details': {'set': ['true', 'false'], 'probs': [1.0, 1.0]},
           'importance_coeff': 0.0,
           'mutation_prob': 0.15,
           'type': 'str',

           'dep_key': 'lt',  # depends on layer type being lstm
           'dep_values': ['lstm', 'gru', 'srnn']}

    params.SetNeuronTraitParameters('ratt', ratt)
    params.SetNeuronTraitParameters('seq', seq)

    # the output activation function and other globals
    at = ['softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    if activations == 'reduced':
        at = ['softmax', 'elu', 'relu', 'tanh', 'linear']
    if activations == 'minimal':
        at = ['softmax', 'tanh', 'linear']
    if mode == 'regression':
        at = ['relu', 'elu', 'linear']
    att = {'details': {'set': at, 'probs': [1.0] * len(at)},
           'importance_coeff': 0.25,
           'mutation_prob': 0.1,
           'type': 'str'}

    # the optimizer
    op = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
    if optimizers == 'reduced':
        op = ['sgd', 'rmsprop', 'nadam']
    opt = {'details': {'set': op, 'probs': [1.0] * len(op)},
           'importance_coeff': 0.05,
           'mutation_prob': 0.05,
           'type': 'str'}

    # learning rate
    if not cont_space:
        s = [.001, .0075, .01, .02]
        p = [1.0] * len(s)
        lrate = {'details': {'set': s, 'probs': p},
                 'importance_coeff': 0.2,
                 'mutation_prob': 0.02,
                 'type': 'floatset', }
    else:
        lrate = {'details': {'min': 0.001, 'max': 0.02, 'mut_power': 0.01, 'mut_replace_prob': 0.15},
                 'importance_coeff': 0.2,
                 'mutation_prob': 0.05,
                 'type': 'float', }

    # the loss
    lo = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
          'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh',
          'categorical_crossentropy', 'kullback_leibler_divergence',
          'poisson', 'cosine_proximity']
    if losses == 'reduced':
        lo = ['mean_squared_error', 'categorical_hinge', 'categorical_crossentropy']
    if mode == 'regression':
        lo = ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_percentage_error',
              'mean_absolute_error', 'logcosh', 'kullback_leibler_divergence',
              'poisson', 'cosine_proximity']
    loss = {'details': {'set': lo, 'probs': [1.0] * len(lo)},
            'importance_coeff': 0.1,
            'mutation_prob': 0.05,
            'type': 'str'}

    params.SetGenomeTraitParameters('outact', att)
    params.SetGenomeTraitParameters('optimizer', opt)
    params.SetGenomeTraitParameters('lrate', lrate)
    params.SetGenomeTraitParameters('loss', loss)


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_true - y_pred) ** 2))


def pretty_name(gt, delim='\n'):
    s=[]
    for k,v in gt.items():
            if k=='lt' and ((v in activation_set) or (v in knownname_set)):
                if (v not in knownname_set):
                    if isinstance(v, float):
                        s.append( '%3.3f' % ( v ) )
                    elif isinstance(v, str):
                        s.append( '%s' % ( v.capitalize() ) )
                    else:
                        s.append('%s' % v)
            else:
                if (k != 'lt') and (v not in knownname_set):
                    if not (v in knownname_set):
                        if isinstance(v, float):
                            s.append( '%s: %3.3f' % (k.capitalize(), v ) )
                        else:
                            s.append( '%s: %s' % (k.capitalize(), v ) )
    if not s: s=[' ']
    return delim.join(s)

def NXGenome2Keras(gr,
                   input_shape, num_outputs,
                   compiled = True,
                   mode = 'classification',
                   regression_metric = 'mean_absolute_error',
                   max_tensor_volume = 1200000):

    # the Keras input layer
    inputs = Input(shape=input_shape, name='Input')

    counter=0
    used=[]

    def getnewname(name=' '):
        while name in used:
            name += ' '
        used.append(name)
        return name

    nl = {}
    nl[1] = inputs  # initialize the input node

    def get_node_shape(node):
        tm = Model(inputs=inputs, outputs=node)
        tm = tm.output_shape[1:]  # drop the batch_size dimension
        return tm

    kiniter = 'glorot_normal'

    for a in list(nx.dfs_postorder_nodes(gr))[::-1][1:]:
        ed = list(gr.in_edges(nbunch=a))  # all incoming edges to this node

        # create the layer
        lt = gr.node[a]

        name = uuname()#pretty_name(lt)
        #name = getnewname(name)

        # activation layers

        if lt['lt'] == 'relu':
            lay = Activation('relu', name=name)
        elif lt['lt'] == 'elu':
            lay = Activation('elu', name=name)
        elif lt['lt'] == 'selu':
            lay = Activation('selu', name=name)
        elif lt['lt'] == 'softmax':
            lay = Activation('softmax', name=name)
        elif lt['lt'] == 'linear':
            lay = Activation('linear', name=name)
        elif lt['lt'] == 'tanh':
            lay = Activation('tanh', name=name)
        elif lt['lt'] == 'softsign':
            lay = Activation('softsign', name=name)
        elif lt['lt'] == 'softplus':
            lay = Activation('softplus', name=name)
        elif lt['lt'] == 'exp':
            lay = Activation('exponential', name=name)
        elif lt['lt'] == 'hsigmoid':
            lay = Activation('hard_sigmoid', name=name)
        elif lt['lt'] == 'sigmoid':
            lay = Activation('sigmoid', name=name)
        elif lt['lt'] == 'leakyrelu':
            lay = LeakyReLU(name=name)
        elif lt['lt'] == 'prelu':
            lay = PReLU(name=name)
        elif lt['lt'] == 'trelu':
            lay = ThresholdedReLU(name=name)

        # other layer types

        elif lt['lt'] == 'dense':
            lay = Dense(lt['size'],
                        kernel_initializer=kiniter,
                        bias_initializer=kiniter,
                        name = name
                        )
        elif lt['lt'] == 'dropout':
            lay = Dropout(lt['dr'], name=name )
        elif lt['lt'] == 'actreg':
            lay = ActivityRegularization(lt['ar1'],
                                         lt['ar2'], name=name
                                         )
        elif lt['lt'] == 'noise':
            lay = GaussianNoise(lt['dr'], name=name )
        elif lt['lt'] == 'dropnoise':
            lay = GaussianDropout(lt['dr'], name=name )
        elif lt['lt'] == 'bnorm':
            lay = BatchNormalization(name=name)
        elif lt['lt'] == 'conv1d':
            lay = Conv1D(lt['filters'],
                         lt['ksize'],
                         padding=lt['padding'],
                         kernel_initializer=kiniter,
                         bias_initializer=kiniter, name=name
                         )
        elif lt['lt'] == 'conv2d':
            lay = Conv2D(lt['filters'],
                         lt['ksize'],
                         kernel_initializer=kiniter,
                         bias_initializer=kiniter, name=name
                         )
        elif lt['lt'] == 'spconv2d':
            lay = SeparableConv2D(lt['filters'],
                                  lt['ksize'],
                                  kernel_initializer=kiniter,
                                  bias_initializer=kiniter, name=name
                                  )
        elif lt['lt'] == 'conv2dt':
            lay = Conv2DTranspose(lt['filters'],
                                  lt['ksize'],
                                  kernel_initializer=kiniter,
                                  bias_initializer=kiniter, name=name
                                  )
        elif lt['lt'] == 'lc1d':
            lay = LocallyConnected1D(lt['filters'],
                                     lt['ksize'],
                                     kernel_initializer=kiniter,
                                     bias_initializer=kiniter, name=name
                                     )
        elif lt['lt'] == 'lc2d':
            lay = LocallyConnected2D(lt['filters'],
                                     lt['ksize'],
                                     kernel_initializer=kiniter,
                                     bias_initializer=kiniter, name=name
                                     )
        elif lt['lt'] == 'maxpool1d':
            lay = MaxPooling1D(pool_size=lt['psize'], name=name )
        elif lt['lt'] == 'maxpool2d':
            lay = MaxPooling2D(pool_size=lt['psize'], name=name )
        elif lt['lt'] == 'avgpool1d':
            lay = AveragePooling1D(pool_size=lt['psize'], name=name )
        elif lt['lt'] == 'avgpool2d':
            lay = AveragePooling2D(pool_size=lt['psize'], name=name )
        elif lt['lt'] == 'gmaxpool1d':
            lay = GlobalMaxPooling1D(name=name)
        elif lt['lt'] == 'gmaxpool2d':
            lay = GlobalMaxPooling2D(name=name)
        elif lt['lt'] == 'gavgpool1d':
            lay = GlobalAveragePooling1D(name=name)
        elif lt['lt'] == 'gavgpool2d':
            lay = GlobalAveragePooling2D(name=name)
        elif lt['lt'] == 'lstm':
            if lt['seq'] == 'true':
                rs = True
            else:
                rs = False
            lay = LSTM(lt['size'],
                       recurrent_activation=lt['ratt'],
                       return_sequences=rs,
                       kernel_initializer=kiniter,
                       bias_initializer=kiniter, name=name
                       )
        elif lt['lt'] == 'gru':
            if lt['seq'] == 'true':
                rs = True
            else:
                rs = False
            lay = GRU(lt['size'],
                      recurrent_activation=lt['ratt'],
                      return_sequences=rs,
                      kernel_initializer=kiniter,
                      bias_initializer=kiniter, name=name
                      )
        elif lt['lt'] == 'srnn':
            if lt['seq'] == 'true':
                rs = True
            else:
                rs = False
            lay = SimpleRNN(lt['size'],
                            return_sequences=rs,
                            kernel_initializer=kiniter,
                            bias_initializer=kiniter, name=name
                            )

        nochangers = ['dense', 'dropout', 'actreg', 'noise', 'dropnoise', 'bnorm']
        need3d = ['conv2d', 'maxpool2d', 'avgpool2d', 'gmaxpool2d', 'gavgpool2d', 'spconv2d', 'conv2dt', 'lc2d']
        need2d = ['conv1d', 'maxpool1d', 'avgpool1d', 'gmaxpool1d', 'gavgpool1d', 'lc1d', 'lstm', 'gru', 'srnn']

        def reshaped_layer(lt, kn, default_dims=2):

            # determine output dimensions
            inshape = get_node_shape(kn)

            tdims = default_dims  # try to keep dimensionality stable
            # so pictures remain pictures as long as possible
            if lt['lt'] in need2d: tdims = 2
            if lt['lt'] in need3d: tdims = 3

            # try to reshape
            if tdims == 1:
                if len(inshape) == 1:
                    # in 1D to 1D we don't need reshape
                    pass
                else:
                    # reshape whatever this is to 1D
                    kn = Reshape((-1,), name=getnewname(' '))(kn)
            elif tdims == 2:
                # case 3D to 2D
                if len(inshape) == 3:
                    kn = Reshape((inshape[0], inshape[1] * inshape[2]), name=getnewname(' '))(kn)
                # case 1D to 2D
                if len(inshape) == 1:
                    kn = Reshape((inshape[0], 1), name=getnewname(' '))(kn)
            elif tdims == 3:
                # case 2D to 3D
                if len(inshape) == 2:
                    kn = Reshape((inshape[0], inshape[1], 1), name=getnewname(' '))(kn)
                # case 1D to 3D
                if len(inshape) == 1:
                    kn = Reshape((inshape[0], 1, 1), name=getnewname(' '))(kn)

            return kn

        def zeropad_1d(node, zeros):
            if zeros == 0: return node
            node = Reshape((-1, 1), name=getnewname(' '))(node)
            node = ZeroPadding1D((0, zeros), name=getnewname(' '))(node)
            node = Reshape((-1,), name=getnewname(' '))(node)
            return node

        def zeropad_2d(node, zeros, axis=0):
            if zeros == 0: return node
            if axis == 0:
                # add 1 zero row to the bottom of 2D
                tm = get_node_shape(node)  # to remember original dims
                node = Reshape((tm[0], tm[1], 1), name=getnewname(' '))(node)
                node = ZeroPadding2D(((0, zeros), (0, 0)), name=getnewname(' '))(node)
                node = Reshape((-1, tm[1]), name=getnewname(' '))(node)
                return node
            elif axis == 1:
                # add 1 zero column to the right of 2D
                tm = get_node_shape(node)  # to remember original dims
                node = Reshape((tm[0], tm[1], 1), name=getnewname(' '))(node)
                node = ZeroPadding2D(((0, 0), (0, zeros)), name=getnewname(' '))(node)
                node = Reshape((tm[0], -1), name=getnewname(' '))(node)
                return node
            else:
                raise ValueError('Invalid axis')

        def zeropad_3d(node, zeros, axis=0):
            if zeros == 0: return node
            if axis == 0:
                # add 1 zero plane to the x axis of 3D
                tm = get_node_shape(node)  # to remember original dims
                node = Reshape((tm[0], tm[1], tm[2], 1), name=getnewname(' '))(node)
                node = ZeroPadding3D(((0, zeros), (0, 0), (0, 0)), name=getnewname(' '))(node)
                node = Reshape((-1, tm[1], tm[2]), name=getnewname(' '))(node)
                return node
            elif axis == 1:
                # add 1 zero plane to the y axis of 3D
                tm = get_node_shape(node)  # to remember original dims
                node = Reshape((tm[0], tm[1], tm[2], 1), name=getnewname(' '))(node)
                node = ZeroPadding3D(((0, 0), (0, zeros), (0, 0)), name=getnewname(' '))(node)
                node = Reshape((tm[0], -1, tm[2]), name=getnewname(' '))(node)
                return node
            elif axis == 2:
                tm = get_node_shape(node)  # to remember original dims
                node = Reshape((tm[0], tm[1], tm[2], 1), name=getnewname(' '))(node)
                node = ZeroPadding3D(((0, 0), (0, 0), (0, zeros)), name=getnewname(' '))(node)
                node = Reshape((tm[0], tm[1], -1), name=getnewname(' '))(node)
                return node
            else:
                raise ValueError('Invalid axis')

        def merge_nodes(lt, nodelist):
            # reshape each node to target shape
            inrs = [get_node_shape(x) for x in nodelist]
            maxinrs = max([len(x) for x in inrs])
            rshaped = [reshaped_layer(lt, x, default_dims=maxinrs) for x in nodelist]
            rs = [get_node_shape(x) for x in rshaped]

            # determine max dimension for each shape
            xsx = np.vstack([np.array(x) for x in rs])
            maxes = np.max(xsx, axis=0)

            nrs = []
            for i, tx in enumerate(rshaped):  # for each tensor
                # it already has the target shape length
                tsl = len(maxes)
                mx = get_node_shape(tx)
                for ax in range(tsl):  # find which axis needs fixing
                    if (lt['mm'] == 'mm_cnc') and ax == tsl - 1:  # skip last dimension for concat
                        continue
                    if mx[ax] < maxes[ax]:  # this axis needs padding
                        if tsl == 1:
                            tx = zeropad_1d(tx, maxes[ax] - mx[ax])
                        if tsl == 2:
                            tx = zeropad_2d(tx, maxes[ax] - mx[ax], axis=ax)
                        if tsl == 3:
                            tx = zeropad_3d(tx, maxes[ax] - mx[ax], axis=ax)
                nrs.append(tx)

            # now the trait can determine whether to concatenate ot do other things
            if lt['mm'] == 'mm_add':
                node = add(nrs)
            elif lt['mm'] == 'mm_sub':
                node = subtract(nrs)
            elif lt['mm'] == 'mm_mul':
                node = multiply(nrs)
            elif lt['mm'] == 'mm_avg':
                node = average(nrs)
            elif lt['mm'] == 'mm_max':
                node = maximum(nrs)
            elif lt['mm'] == 'mm_min':
                node = minimum(nrs)
            elif lt['mm'] == 'mm_cnc':
                node = concatenate(nrs)
            else:
                raise RuntimeError('Unknown mm trait')

            return node

        if len(ed) == 1:
            # just one input, connect to lay
            kn = nl[ed[0][0]]
            kn = reshaped_layer(lt, kn, default_dims=len(get_node_shape(kn)))

            lay = lay(kn)
        else:
            # many other nodes input to this one, try to merge their outputs
            incs = [nl[inc] for inc, outg in ed]
            conc = merge_nodes(lt, incs)
            kn = conc
            lay = lay(kn)

        # check if the output has 0 or negative dimensions somewhere
        # if so, fix it by padding zeros
        # note: might be quicker to just have it throw an exception, but do this after debug
        """
        tv = get_node_shape(lay)
        if any([x < 1 for x in tv]):
            return None
        # also prevent too big tensors
        #if functools.reduce(operator.mul, tv, 1) > self.max_tensor_volume:
        #    return None
        """

        nl[a] = lay

    # just in case do it again here
    """
    tv = get_node_shape(nl[2])
    if any([x < 1 for x in tv]):
        return None
    # also prevent too big tensors
    if functools.reduce(operator.mul, tv, 1) > self.max_tensor_volume:
        return None
    """

    # the output is yet another dense layer, but we specify the size
    # see if requires Flatten()
    if len(get_node_shape(nl[2])) > 1:
        # needs Flatten()
        out = Dense(num_outputs,
                    kernel_initializer=kiniter,
                    bias_initializer=kiniter,
                    name='Output',
                    activation=gr.genome_traits['outact'])(Flatten()(nl[2]),)
    else:
        # doesn't
        out = Dense(num_outputs,
                    kernel_initializer=kiniter,
                    bias_initializer=kiniter,
                    name='Output',
                    activation=gr.genome_traits['outact'])(nl[2])

    model = Model(inputs=inputs, outputs=out)

    mem_needed = int(np.sum(
        [np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape]))
                           for l in model.layers]))
    if mem_needed > max_tensor_volume:
        return None

    if compiled:
        lrate = gr.genome_traits['lrate']
        opt = SGD(lr=lrate)
        if gr.genome_traits['optimizer'] == 'sgd':
            opt = SGD(lr=lrate)
        if gr.genome_traits['optimizer'] == 'rmsprop':
            opt = RMSprop(lr=lrate)
        if gr.genome_traits['optimizer'] == 'adagrad':
            opt = Adagrad(lr=lrate)
        if gr.genome_traits['optimizer'] == 'adadelta':
            opt = Adadelta(lr=lrate)
        if gr.genome_traits['optimizer'] == 'adam':
            opt = Adam(lr=lrate)
        if gr.genome_traits['optimizer'] == 'adamax':
            opt = Adamax(lr=lrate)
        if gr.genome_traits['optimizer'] == 'nadam':
            opt = Nadam(lr=lrate)

        if mode != 'regression':
            model.compile(loss=gr.genome_traits['loss'], optimizer=opt, metrics=['accuracy'])
        else:
            if regression_metric=='rmse':
                model.compile(loss=gr.genome_traits['loss'],
                              optimizer=opt, metrics=[rmse])
            else:
                model.compile(loss=gr.genome_traits['loss'],
                              optimizer=opt, metrics=[regression_metric])

    return model


#dmode = None
def decide (x):
    return x
    #global dmode
    #if mode:
    #    return x if mode != 'regression' else 1000 - x
    #else:
    #    return x if dmode != 'regression' else 1000 - x


def Trial(gr, x, py, gtx, gtpy,
          patience=8,
          max_epochs=50,
          batch_size=128,
          max_tensor_volume=1200000,
          mode='classification',
          regression_metric='mean_absolute_error'):

    x_shape = x.shape[1:]
    ys = py.shape[1]

    model = NXGenome2Keras(gr, x_shape, ys,
                           mode=mode,
                           max_tensor_volume=max_tensor_volume,
                           regression_metric=regression_metric)

    #if mode != 'regression':
    es = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience, verbose=0, mode='max')
    #else:
    #    es = EarlyStopping(monitor='val_'+regression_metric, min_delta=0, patience=patience, verbose=0, mode='min')
    tnan = TerminateOnNaN()

    # split into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x, py, test_size=0.2,
                                                        shuffle=True,
                                                        random_state=rnd.randint(0, 10000000))

    # train model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=max_epochs, verbose=0,
                        validation_data=(x_test, y_test), callbacks=[es])

    #if mode != 'regression':
        # if 'val_acc' in history.history.keys():
    hs = np.array(history.history['val_acc'])
    vacc = np.max(hs)
    # else:
    #    vacc = 0.0
    #else:
    #    # if 'val_mae' in history.history.keys():
    #    hs = np.array(history.history['val_'+regression_metric])
    #    vacc = np.min(hs)
    # else:
    #    vacc = 0.0

    if gtx is not None:
        p = model.predict(gtx)
        p = np.argmax(p, axis=1)
        pvacc = accuracy_score(np.argmax(gtpy,axis=1), p)
    else:
        pvacc = 0

    return vacc, pvacc


def Evaluate(args):
    idx, gr, x, py, gtx, gtpy, trials, patience, max_epochs, \
    batch_size, max_tensor_volume, mode, regression_metric = args

    results = []
    r2r = []
    for t in range(trials):
        try:
            r, r2 = Trial(gr, x, py, gtx, gtpy,
                      patience=patience,
                      max_epochs=max_epochs,
                      batch_size=batch_size,
                      max_tensor_volume=max_tensor_volume,
                      mode=mode,
                      regression_metric=regression_metric)
            if trials>0:
                print(f'genome: #{idx} { gr2kn(gr) + [gr.genome_traits] }, trial #{t}: {r}')
                print()

            if np.isnan(r) or np.isinf(r):
                return idx, 0.0, 0.0, results

        except Exception as ex:
            print(ex)
            return idx, 0.0, 0.0, results

        results.append(r)
        r2r.append(r2)

    #if mode == 'regression':
    #    fs = (1000.0 - np.mean(results)) - 0.5*np.std(results)
    #else:
    fs = np.mean(results)
    fs2 = np.mean(r2r)

    if np.isnan(fs) or np.isinf(fs) or (fs < 0.0):
        fs = 0.0
        fs2 = 0.0

    print(f'genome: #{idx} { gr2kn(gr) + [gr.genome_traits] } mean: {np.mean(results)}')
    print()
    print()
    return idx, fs, fs2, results


def LabelGenomeGraph(gr):
    for n, node in gr.nodes(data=1):
        s = []
        for k, v in sorted(node.items()):
            if isinstance(v, float):
                s += ['%3.2f' % v]
            else:
                s += [str(v)]
        s = '\n'.join(s)
        node['label'] = s

    # in our use case only
    gr.node[1]['label'] = 'input'


def GenomeViz(genome, x_shape, ys):
    gr = NEAT.Genome2NX(genome)
    model = NXGenome2Keras(gr, x_shape, ys)
    uu = uuname()
    plot_model(model, show_shapes=False, show_layer_names=True, to_file=f'/tmp/model_{uu}.png')
    p = pyplot.imread(open(f'/tmp/model_{uu}.png', 'rb'))
    os.remove(f'/tmp/model_{uu}.png')
    return p


verbose_constraints = False

def fails_constraints(genome):
    try:
        if verbose_constraints: print('Validating a genome..', end='')
        gr = NEAT.Genome2NX(genome)

        # topology constraints
        n = list(gr.nodes(data=True))
        ndic = dict(n)

        a = dict(nx.bfs_successors(gr, 1))
        x = [(xx, a[xx][0]) for xx in a]

        # sort n also
        n = [xx for xx, yy in x][1:]
        n = [(xx, ndic[xx]) for xx in n] + [(x[-1][1], ndic[x[-1][1]])]

        x = [(xx, a[xx][0]) for xx in a][1:]
        x = [((xx, ndic[xx]), (yy, ndic[yy])) for xx, yy in x]

        # don't allow connected pooling layers in series
        for prev, nxt in x:
            if 'pool' in prev[1]['lt'] and 'pool' in nxt[1]['lt']:
                if verbose_constraints: print('Failed (connected pooling layers in series).')
                return True

        # don't allow connected bnorm layers in series
        for prev, nxt in x:
            if 'bnorm' in prev[1]['lt'] and 'bnorm' in nxt[1]['lt']:
                if verbose_constraints: print('Failed (connected bnorm layers in series).')
                return True

        # don't allow connected actreg layers in series
        for prev, nxt in x:
            if 'actreg' in prev[1]['lt'] and 'actreg' in nxt[1]['lt']:
                if verbose_constraints: print('Failed (connected actreg layers in series).')
                return True

        # don't allow pooling layers to be first
        if 'pool' in n[0][1]['lt']:
            if verbose_constraints: print('Failed (pooling layers are first).')
            return True

        # don't allow bnorm layers to be first
        if 'bnorm' in n[0][1]['lt']:
            if verbose_constraints: print('Failed (bnorm layers are first).')
            return True

        # don't allow actreg layers to be first
        if 'actreg' in n[0][1]['lt']:
            if verbose_constraints: print('Failed (actreg layers are first).')
            return True

        # don't allow convolutional layers to be last
        if 'conv' in n[-1][1]['lt']:
            if verbose_constraints: print('Failed (convolutional layers are last).')
            return True

        # don't allow connected dropout layers in series
        for prev, nxt in x:
            if 'drop' in prev[1]['lt'] and 'drop' in nxt[1]['lt']:
                if verbose_constraints: print('Failed (connected dropout layers in series).')
                return True

        # don't allow dropout layers to be first
        if 'drop' in n[0][1]['lt']:
            if verbose_constraints: print('Failed (dropout layers are first).')
            return True

        # don't allow dense layers to be first
        # note: this is valid only for 2D+ input spaces
        # if 'dense' in n[0][1]['lt']:
        #    print('Failed (dense layers are first).')
        #    return True

        #  don't allow softmax except at the end
        for indx in range(len(n[:-1])):
            if 'act' in n[indx][1].keys():
                if 'softmax' in n[indx][1]['act']:
                    if verbose_constraints: print('Failed (softmax not at the end).')
                    return True

        # passed all topology constraints

        # test if we can compile and run it
        # model = NXGenome2Keras(gr, x_shape, ys)
        # if model is None:
        #    if verbose_constraints: print("Failed (couldn't build model).")
        #    return True

        # model.compile(loss='mse', optimizer=SGD(), metrics=['accuracy'])
        # model.predict(numpy.zeros((1,)+x_shape))

        if verbose_constraints: print('OK')
        return False

    except KeyboardInterrupt:
        raise

    except Exception as ex:
        if verbose_constraints: print(f'Failed (unknown reason: {ex}).')
        return True


# to ease conversion
def gr2kn(gr):
    n = gr.nodes(data=True)
    ndic = dict(n)
    a = dict(nx.bfs_successors(gr, 1))
    x = [(xx, a[xx][0]) for xx in a]
    n = [xx for xx, yy in x][1:]
    n = [(xx, ndic[xx]) for xx in n] + [(x[-1][1], ndic[x[-1][1]])]
    x = [(xx, a[xx][0]) for xx in a][1:]
    x = [((xx, ndic[xx]), (yy, ndic[yy])) for xx, yy in x]
    kn = [x[1] for x in n]
    for d in kn:
        if 'mm' in d:
            del d['mm']
    return kn


def genome2kn(g):
    gr = NEAT.Genome2NX(g)
    return gr2kn(gr)


def get_model_from_genome(genome, x_shape, ys,
                          mode='classification', regression_metric='mean_absolute_error'):
    return NXGenome2Keras(NEAT.Genome2NX(genome), x_shape, ys,
                          mode=mode,
                          regression_metric=regression_metric)

def pretty_gtraits(gt, delim='\n'):
    s = []
    for k,v in gt.items():
        if isinstance(v, float):
            s.append( '%s: %3.3f' % (k, v) )
        else:
            s.append( '%s: %s' % (k, v) )
    return delim.join(s)



def plot_progress(evhist, pop, x_shape, ys, display_whole_pop=0, display_max_species=6):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3))
    print('\nOverall Stats:')
    # Display population with species colored
    ax1.set_title('Population')
    plot_pop(ax1, pop)
    # Display best fitness history
    ax2.set_title('Best fitness history')
    eee = np.array(evhist)
    eee = eee[eee != 0.0]
    ax2.plot(eee)
    f.subplots_adjust(hspace=0)
    f.tight_layout()
    plt.show()

    def poplen(pop):
        return sum([len(x.Individuals) for x in pop.Species])

    # And display the best genome
    f, ax1 = plt.subplots(1, figsize=(4, 6))
    print('Best Genome:')
    ax1.imshow(GenomeViz(pop.GetBestGenome(), x_shape, ys))
    ax1.set_title('%3.6f' % pop.GetBestGenome().GetFitness())
    ax1.set_xlabel('\n'.join([pretty_gtraits(x, delim=', ') for x in genome2kn(pop.GetBestGenome())]) + '\n\n'
                   + pretty_gtraits(pop.GetBestGenome().GetGenomeTraits()))
    #f.tight_layout()
    plt.show()
    # Also print the best genome
    #print('\n'.join([str(x) for x in genome2kn(pop.GetBestGenome())]))

    if display_whole_pop and (len(pop.Species)>1):
        genomes = [x.Individuals[0] for x in pop.Species][0:display_max_species]
        f, axes = plt.subplots(1, len(genomes), figsize=(len(genomes) * 4.5, 14))
        print('Species Representatives:')
        for i,(ax, g) in enumerate(zip(axes, genomes)):
            ax.imshow(GenomeViz(g, x_shape, ys))
            ax.set_title('%3.6f | %3.2f%%' % (g.GetFitness(),
                                              (len(pop.Species[i].Individuals)/poplen(pop))*100 ))
            ax.set_xlabel('\n'.join([pretty_gtraits(x, delim=', ') for x in genome2kn(g)]) + '\n\n'
                          + pretty_gtraits(g.GetGenomeTraits()))
        f.tight_layout()
        plt.show()
    #else:


def plot_pop(ax1, pop):
    ti = 0
    for sp in pop.Species:
        ax1.bar(range(ti, ti + len(sp.Individuals)), [ (x.GetFitness()) if x.IsEvaluated() else -250.0
                                                      for x in sp.Individuals
                                                      ],
                width=1.0,
                color=(sp.Red / 255.0, sp.Green / 255.0, sp.Blue / 255.0))
        ti += len(sp.Individuals)


def variance_report(vr):
    global dmode
    if vr:
        vr = [np.clip(x, -10000, 10000) for x in vr]
        return ', '.join(['%3.3f' % x for x in [min(vr), max(vr)]]
                         + ['%3.5f' % np.mean(vr), '%3.5f' % np.std(vr)])
    else:
        return ''


def deepneat_run(dx, dy, generations=50, evaluations=300, population=64, num_layers=0,
                 domain_type='simple', activations='all', optimizers='reduced', losses='reduced',
                 mode='classification', regression_metric='mean_absolute_error', cluster='192.168.0.108:8786',
                 cont_space=True, trials=5, patience=8, max_epochs=50, batch_size=128,
                 max_tensor_volume=2500000, ground_truth=None,
                 display_whole_pop=1, display_progress_each=50, display_max_species=8,
                 archive=False, initeval=0, parallel=1, rtneat=1, viz=1):

    once=False
    global dmode
    dmode=mode

    x, py, x_shape, ys = process_xy(dx, dy, mode)

    if ground_truth is not None:
        gtx, gty = ground_truth
        gtx, gtpy, _, _ = process_xy(gtx, gty, mode)
    else:
        gtx, gtpy = None, None

    init_traits(params,
                cont_space = cont_space,
                domain_type=domain_type,
                activations=activations,
                optimizers=optimizers,
                losses=losses,
                mode=mode)
    params.PopulationSize = population
    params.ArchiveEnforcement = archive
    params.CustomConstraints = fails_constraints
    g = NEAT.Genome(0, 1, 1, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 1, params, num_layers, 1)
    pop = NEAT.Population(g, params, True, 1.0, rnd.randint(0, 100000000))

    if parallel:
        import time
        st = time.time()
        print('Connecting..')
        if cluster:
            client = Client(cluster)
        else:
            client = Client()

        # push the data to the cluster
        print('Scattering data to cluster..')
        fx = client.scatter(x, broadcast=True, direct=True)
        fpy = client.scatter(py, broadcast=True, direct=True)
        fgtx = client.scatter(gtx, broadcast=True, direct=True)
        fgtpy = client.scatter(gtpy, broadcast=True, direct=True)
        print(f'done in {time.time()-st} seconds')
    else:
        fx, fpy = None, None
        fgtx, fgtpy = None, None
        client = None

    evhist = []
    best_gs = []
    best_ever=-1

    import time

    try:
        # main evolution loop
        if not rtneat:
            for generation in range(generations):
                print('============================================================')
                print("Generation:", generation)

                now = time.time()
                genome_list = []
                for s in pop.Species:
                    for i in s.Individuals:
                        genome_list.append(i)

                print('Population size:', len(genome_list), 'Species:', len(pop.Species))
                sys.stdout.flush()

                # turn them into NX networks
                grlist = [NEAT.Genome2NX(x) for x in genome_list]

                if parallel:
                    args = list(zip([x.GetID() for x in NEAT.GetGenomeList(pop)],
                                    grlist,
                                    [fx]*len(grlist),
                                    [fpy]*len(grlist),

                                    [fgtx] * len(grlist),
                                    [fgtpy] * len(grlist),

                                    [trials]*len(grlist),
                                    [patience]*len(grlist),
                                    [max_epochs]*len(grlist),
                                    [batch_size]*len(grlist),
                                    [max_tensor_volume]*len(grlist),
                                    [mode]*len(grlist),
                                    [regression_metric]*len(grlist)
                                    ))
                else:
                    args = list(zip([x.GetID() for x in NEAT.GetGenomeList(pop)],
                                    grlist,
                                    [x]*len(grlist),
                                    [py]*len(grlist),

                                    [gtx] * len(grlist),
                                    [gtpy] * len(grlist),

                                    [trials] * len(grlist),
                                    [patience] * len(grlist),
                                    [max_epochs] * len(grlist),
                                    [batch_size] * len(grlist),
                                    [max_tensor_volume] * len(grlist),
                                    [mode] * len(grlist),
                                    [regression_metric]*len(grlist)
                                    ))

                # evaluate all individuals
                fitnesses = {}
                vrs = {}
                bar = ProgressBar(initial_value=0, max_value=len(grlist))
                if parallel:
                    cp = [client.submit(Evaluate, x) for x in args]
                    for i,ftr in enumerate(as_completed(cp)):
                        idx, fitness, freal, vr = ftr.result()
                        fitnesses[idx] = fitness
                        vrs[idx] = vr
                        bar.update(i)
                else:
                    for indx, i in enumerate([x.GetID() for x in NEAT.GetGenomeList(pop)]):
                        idx, fitness, freal, vr = Evaluate(args[indx])
                        fitnesses[i] = fitness
                        vrs[indx] = vr
                        bar.update(indx)
                bar.finish()

                for k,v in fitnesses.items():
                    pop.AccessGenomeByID(k).SetFitness(v)
                    pop.AccessGenomeByID(k).SetEvaluated()

                # get best fitness in population and print it
                fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
                best = max(fitness_list)
                bestidx = np.argmax(fitness_list)
                evhist.append( (best))
                if best > best_ever:
                    print('NEW RECORD!')
                    print('Generations:', generation,
                          'Fitness:',  (best),
                          'Variance (min/max/mean/std):', '')
                    best_gs.append(pop.GetBestGenome())
                    best_ever = best
                    if viz:
                        plot_progress(evhist, pop, x_shape, ys,
                                      display_whole_pop=display_whole_pop,
                                      display_max_species=display_max_species)

                print()
                print('Best result: %3.4f' % (float(best)))
                print("Evaluation took", '%3.2f' % (time.time() - now), "seconds.")
                pop.Epoch()
        else:
            # rtNEAT code
            if initeval:
                print('============================================================')
                print("Please wait for the initial evaluation to complete.")
                now = time.time()
                genome_list = []
                for s in pop.Species:
                    for i in s.Individuals:
                        genome_list.append(i)
                sys.stdout.flush()
                # turn them into NX networks
                grlist = [NEAT.Genome2NX(x) for x in genome_list]
                if parallel:
                    args = list(zip([x.GetID() for x in NEAT.GetGenomeList(pop)],
                                    grlist,
                                    [fx]*len(grlist),
                                    [fpy]*len(grlist),

                                    [fgtx] * len(grlist),
                                    [fgtpy] * len(grlist),

                                    [trials] * len(grlist),
                                    [patience] * len(grlist),
                                    [max_epochs] * len(grlist),
                                    [batch_size] * len(grlist),
                                    [max_tensor_volume] * len(grlist),
                                    [mode] * len(grlist),
                                    [regression_metric]*len(grlist)

                                    ))
                else:
                    args = list(zip([x.GetID() for x in NEAT.GetGenomeList(pop)],
                                    grlist,
                                    [x]*len(grlist),
                                    [py]*len(grlist),

                                    [gtx] * len(grlist),
                                    [gtpy] * len(grlist),

                                    [trials] * len(grlist),
                                    [patience] * len(grlist),
                                    [max_epochs] * len(grlist),
                                    [batch_size] * len(grlist),
                                    [max_tensor_volume] * len(grlist),
                                    [mode] * len(grlist),
                                    [regression_metric]*len(grlist)

                                    ))

                # evaluate all individuals
                fitnesses = [0] * len(grlist)
                vrs = [None] * len(grlist)
                bar = ProgressBar(initial_value=0, max_value=len(grlist))
                if parallel:
                    cp = [client.submit(Evaluate, x) for x in args]
                    for i,ftr in enumerate(as_completed(cp)):
                        idx, fitness, freal, vr = ftr.result()
                        fitnesses[idx] = fitness
                        vrs[idx]=vr
                        bar.update(i)
                else:
                    for i in range(len(grlist)):
                        idx, fitness, freal, vr = Evaluate(args[i])
                        fitnesses[i] = fitness
                        vrs[i]=vr
                        bar.update(i)
                bar.finish()

                NEAT.ZipFitness(genome_list, fitnesses)
                print("Evaluation took", '%3.2f' % (time.time() - now), "seconds.")
            else:
                # No initial evaluation, start off with small random fitness
                for s in pop.Species:
                    for i in s.Individuals:
                        i.SetFitness(np.random.rand()/1000)
                        #if not parallel: i.SetEvaluated()
                if parallel: # but send everyone for evaluation still
                    grlist = [NEAT.Genome2NX(x) for x in NEAT.GetGenomeList(pop)]
                    args = list(zip([x.GetID() for x in NEAT.GetGenomeList(pop)],
                                    grlist,
                                    [fx]*len(grlist),
                                    [fpy]*len(grlist),

                                    [fgtx] * len(grlist),
                                    [fgtpy] * len(grlist),

                                    [trials] * len(grlist),
                                    [patience] * len(grlist),
                                    [max_epochs] * len(grlist),
                                    [batch_size] * len(grlist),
                                    [max_tensor_volume] * len(grlist),
                                    [mode] * len(grlist),
                                    [regression_metric]*len(grlist)

                                    ))
                    cp = [client.submit(Evaluate, x) for x in args]

            # get best fitness in population and print it
            fitness_list = [x.GetFitness() for x in NEAT.GetGenomeList(pop)]
            best = max(fitness_list)
            if initeval: evhist.append( (best))
            if best > best_ever:
                print('NEW RECORD!')
                print('Evaluations:', 0,
                      'Fitness:',  (best),
                      'Species:', len(pop.Species))
                best_gs.append(pop.GetBestGenome())
                best_ever = best
                if viz:
                    plot_progress(evhist, pop, x_shape, ys,
                                  display_whole_pop=display_whole_pop,
                                  display_max_species=display_max_species)

            print('============================================================')
            print('rtNEAT phase')
            print('============================================================')

            format_custom_text = FormatCustomText('CTresh: %(ctr).3f Species: %(sp)d Last fitness: %(fitness).3f Variance (min/max/mean/std): %(vr)s',
                dict(
                    sp=0,
                    fitness=0.0,
                    vr=str([]),
                    ctr=0.0,
                ),)

            widgets = ['Evaluated: ', Counter('%(value)d'), ' ', format_custom_text,
                       ' (', Timer(), ') ', RotatingMarker(), FileTransferSpeed(unit='individuals') ]
            bar = ProgressBar(widgets=widgets, initial_value=0, max_value=evaluations)

            if not parallel:
                # continuous loop
                for i in range(evaluations):
                    bar.update(i)

                    # get the new baby
                    old = NEAT.Genome()
                    baby = pop.Tick(old)

                    # evaluate it
                    f = Evaluate((baby.GetID(),
                                  NEAT.Genome2NX(baby),
                                  x,
                                  py,
                                  gtx,
                                  gtpy,
                                  trials,
                                  patience,
                                  max_epochs,
                                  batch_size,
                                  max_tensor_volume,
                                  mode,
                                  regression_metric
                                  ))

                    # recalculate real fitness here

                    baby.SetFitness(f[1])
                    baby.SetEvaluated()

                    # get best fitness in population and print it
                    evhist.append( (f[1]))
                    if f[1] > best_ever:
                        print('NEW RECORD!')
                        print('Evaluations:', i,
                              'Fitness:',  (f[1]),
                              'GT:', (f[2]),
                              'Variance (min/max/mean/std):', variance_report(f[3]))
                        best_gs.append(pop.GetBestGenome())
                        best_ever = f[1]
                        if viz:
                            plot_progress(evhist, pop, x_shape, ys,
                                          display_whole_pop=display_whole_pop,
                                          display_max_species=display_max_species)
                    elif (i%display_progress_each)==0:
                        print('Population checkup')
                        plot_progress(evhist, pop, x_shape, ys,
                                      display_whole_pop=display_whole_pop,
                                      display_max_species=display_max_species)


                    format_custom_text.update_mapping(fitness= (f[1]),
                                                      vr=variance_report(f[2]),
                                                      sp=len(pop.Species),
                                                      ctr=params.CompatTreshold)
            else:
                """
                # parallel continious loop
                seq = as_completed(cp)
                i=0
                olds = []
                under_evaluation_now = len(cp)
                for f in seq:
                    if i >= evaluations:
                        [x.cancel() for x in cp]
                        break
                    if i > 0: bar.update(i)
                    i += 1

                    # get result from evaluation
                    idx, fitness, vr = f.result()
                    if idx not in olds:
                        # set that individual's fitness
                        pop.AccessGenomeByID(idx).SetFitness( (fitness))
                        pop.AccessGenomeByID(idx).SetEvaluated()
                        under_evaluation_now -= 1
                    else:
                        print(f'Fitness of genome #{idx} was not set.')
                    format_custom_text.update_mapping(fitness= (fitness),
                                                      vr=variance_report(vr),
                                                      sp=len(pop.Species),
                                                      ctr=params.CompatTreshold)

                    # get best fitness in population and print it
                    evhist.append( (fitness))
                    if fitness > best_ever:
                        print('NEW RECORD!')
                        print('Evaluations:', i,
                              'Fitness:',  (fitness),
                              'Variance (min/max/mean/std):', variance_report(vr))
                        best_gs.append(pop.GetBestGenome())
                        best_ever = fitness
                        if viz:
                            plot_progress(evhist, pop, x_shape, ys,
                                          display_whole_pop=display_whole_pop,
                                          display_max_species=display_max_species)
                    elif (i%display_progress_each)==0:
                        print('Population checkup')
                        plot_progress(evhist, pop, x_shape, ys,
                                      display_whole_pop=display_whole_pop,
                                      display_max_species=display_max_species)

                    # create new baby and add to list of tasks
                    # only add new babies if the population under evaluation is below N%
                    if under_evaluation_now <= (population*(1/3)):
                        if not once:
                            once=True
                            print(f'rtNEAT reproduction cycle started at evaluation #{i}')

                        old = NEAT.Genome()
                        baby = pop.Tick(old)
                        if old.GetID() != -1:
                            olds.append(old.GetID())
                        newf = client.submit(Evaluate, (baby.GetID(),
                                                        NEAT.Genome2NX(baby),
                                                        fx,
                                                        fpy,
                                                        trials,
                                                        patience,
                                                        max_epochs,
                                                        batch_size,
                                                        max_tensor_volume,
                                                        mode,
                                                        regression_metric
                                                        ))
                        seq.add(newf)
                        under_evaluation_now += 1
                """
                verbose=1
                # parallel continious loop
                seq = as_completed(cp)
                i = 0
                olds = []
                under_evaluation_now = len(cp)
                for f in tqdm(seq):
                    if i >= evaluations:
                        [x.cancel() for x in cp]
                        break
                    if i > 0: bar.update(i)
                    i += 1

                    # get result from evaluation
                    idx, fitness, freal, rsx = f.result()
                    if 1:
                        if idx not in olds:
                            # set that individual's fitness
                            pop.AccessGenomeByID(idx).SetFitness(fitness)
                            pop.AccessGenomeByID(idx).SetEvaluated()
                            under_evaluation_now -= 1

                            thegenome = pop.AccessGenomeByID(idx)

                            #  apply the local search back to genome
                            # if use_local_search:
                            #    for qx,w in zip(thegenome.LinkGenes,rsx): qx.Weight = w

                            # get best fitness in population and print it

                            #fhist.append(decide(fitness))
                            if fitness > best_ever:
                                if verbose:
                                    print('NEW RECORD!')
                                    print('Evaluations:', i,
                                          'Fitness:', decide(fitness),
                                          'On ground truth:', decide(freal),)

                                best_gs.append([NEAT.Genome2NX(thegenome, with_weights=1),
                                                thegenome.GetLinkTraits(1),
                                                thegenome.GetNeuronTraits(),
                                                rsx
                                                ])

                                evhist.append(fitness)

                                if verbose:
                                    print(f'-------- nodes: {thegenome.NumNeurons()} -------- ')
                                    print('\n'.join([str(x) for x in thegenome.GetNeuronTraits() if x[1] != 'input'][
                                                    0:8]))
                                    print(f'-------- links: {thegenome.NumLinks()} -------- ')
                                    print('\n'.join(
                                        [str(x) for x in thegenome.GetLinkTraits(1)][0:8]))
                                    print('----------------------- ')

                                best_ever = fitness
                                #if viz and verbose:
                                #    species_display(pop)
                            #elif (i % display_pop_each) == 0:
                            #    if viz and verbose:
                            #        species_display(pop)

                        else:
                            if verbose:
                                print(f'Fitness of genome #{idx} was not set.')

                    # except Exception as ex:
                    #    print(ex)
                    if verbose:
                        format_custom_text.update_mapping(fitness=decide(fitness),
                                                          sp=len(pop.Species),
                                                          ctr=pop.Parameters.CompatTreshold)

                    # apply exploration pressure to species
                    #if penalize_stangation:
                    #    maxfit = max([x.GetFitness() for x in NEAT.GetGenomeList(pop)])
                    #    for s in pop.Species:
                    #        if s.EvalsNoImprovement > penalize_stagnation_evals:
                    #            bf = max([x.GetFitness() for x in s.Individuals])
                    #            # make an exception if that species contains the best genome so far
                    #            if bf < maxfit:
                    #                for ind in s.Individuals:
                    #                    ind.SetFitness(0.000000001)

                    #if all([x.EvalsNoImprovement > max_stagnation for x in pop.Species]):
                    #    break

                    # create new baby and add to list of tasks
                    # only add new babies if the population under evaluation is below N%
                    if under_evaluation_now <= (population * (1 / 3)):
                        if not once:
                            once = True
                            if verbose:
                                print(f'rtNEAT reproduction cycle started at evaluation #{i}')

                        old = NEAT.Genome()
                        baby = pop.Tick(old)

                        if old.GetID() != -1:
                            olds.append(old.GetID())

                        newf = client.submit(Evaluate, (baby.GetID(),
                                                        NEAT.Genome2NX(baby),
                                                        fx,
                                                        fpy,
                                                        fgtx,
                                                        fgtpy,
                                                        trials,
                                                        patience,
                                                        max_epochs,
                                                        batch_size,
                                                        max_tensor_volume,
                                                        mode,
                                                        regression_metric
                                                        ))
                        seq.add(newf)
                        under_evaluation_now += 1


    except KeyboardInterrupt:
        print('Interrupted by user.')

    if parallel:
        try:
            client.cancel(seq)
        except:
            pass
        client.restart(timeout=600)
        print('Cluster restarted.')

    return best_gs


def process_xy(dx, dy, mode):
    x = dx
    if mode != 'regression':
        py = to_categorical(dy)
    else:
        py = dy
    if len(py.shape) == 1:
        py = py.reshape(-1, 1)
    x_shape = dx.shape[1:]
    ys = py.shape[1]
    return x, py, x_shape, ys

