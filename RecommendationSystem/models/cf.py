# This script implements the item-item collaborative filtering algorithm. It uses the cosine similarity to find the similarity between items. The algorithm is implemented in the CF class.

from utils.config import np, pd
from preprocessing.cache import load_from_pickle, save_to_pickle

class CollaborativeFiltering:
    """
    The class to implement the item-item collaborative filtering algorithm
    """
    def __init__(self):
        self.bias = None
        self.similarity = None        

    def calc_bias(self):
        """
        The function to calculate the bias for the items
        Without the factorization, the bias is calculated as:
        b_i = μ + b_x + b_i
        """
        # Load the statistical data
        stat_data = load_from_pickle('data/cache/pkls/train_statistics.pkl')
        μ = stat_data['μ']
        b_x = stat_data['b_x']
        b_i = stat_data['b_i']
        self.bias = μ + b_x + b_i

