from utils.config import np, pd
from utils.config import train_data_path, test_data_path, attributes_path, results_path
from utils.config import item_idx, users, items
from utils.helpers import generate_index_map
from cache import load_from_pickle

if __name__ == '__main__' :
    """
    The main function to preprocess the data
    """

    print('----------------- Preprocessing Train Data -----------------')
    # Create the index map for the items
    print("Creating the index map for the items...")
    generate_index_map(train_data_path)
    print("Done")
    # Load the index map
    item_idx = load_from_pickle('data/cache/item_idx.pkl')

