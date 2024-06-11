# This script defines the configuration of the project. It imports the necessary libraries and sets the global variables.

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

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
train_data_stats_path = 'data/cache/train_data_stats.txt'
train_data_ratings_distribution_path = 'data/cache/train_data_ratings_distribution.png'