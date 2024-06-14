# This script defines the configuration of the project. It imports the necessary libraries and sets the global variables.

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
import IncSVD.EvolvingMatrix as EM
import logging
from tqdm import tqdm



# Global variables
item_idx = defaultdict()
users = defaultdict()
items = defaultdict()

# Colors and styles
colors = ['#d28b91', # red
          '#90c29e', # green
          '#e1ac91', # orange
          '#8aa1c9']  # blue
plt.rcParams['axes.facecolor'] = '#eaeaf2'
plt.rcParams['axes.unicode_minus'] = False  

# File paths
train_data_path = 'data/train.txt'
test_data_path = 'data/test.txt'
attributes_path = 'data/itemAttribute.txt'
results_path = 'data/ResultForm.txt'
train_data_stats_path = 'data/cache/txts/train_data_stats.txt'
train_data_ratings_distribution_path = 'data/cache/pics/train_data_ratings_distribution.png'
test_data_stats_path = 'data/cache/txts/test_data_stats.txt'
attributes_stats_path = 'data/cache/txts/attributes_stats.txt'
attributes_distribution_path = 'data/cache/pics/attributes_distribution.png'
