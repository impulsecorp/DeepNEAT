import sys
sys.path.append('/home/peter/code/projects')
from aidevutil import *
from deepneat import deepneat_run

dx = np.load('dx.npy')
dy = np.load('dy.npy')

best_gs = deepneat_run(dx, dy)


















