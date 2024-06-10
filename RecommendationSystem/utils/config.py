# This script defines the configuration of the project. It imports the necessary libraries and sets the global variables.

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

# Global variables
item_idx = defaultdict()
users = defaultdict()
items = defaultdict()



# File paths
train_data_path = 'data/train.txt'
test_data_path = 'data/test.txt'
attributes_path = 'data/itemAttribute.txt'
results_path = 'data/ResultForm.txt'
