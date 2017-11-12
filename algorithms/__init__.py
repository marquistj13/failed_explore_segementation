import logging

from apcm import apcm
from upcm import upcm
# Initialize bandwidth via KMeans,
from npcm_plot import npcm_plot # Initialize bandwidth via KMeans, without the plot of the sigma_v curve.

from npcm_plot_fcm_flag import npcm_plot_fcm

# note that the above algorithms is designed to work with the animation feature
# please use the algorithms from pcm_based_algorithms for regular use

from metrics import calculate_rand_index, calculate_purity, calculate_mean_euclidean_distances
from pcm_based_algorithms import pcm_algorithm,apcm_algorithm,upcm_algorithm,npcm_algorithm,npcm_algorithm_fcm

# log setup
logger = logging.getLogger('algorithm')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logging.log', 'w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
