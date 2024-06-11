from utils.config import train_data_path, test_data_path, attributes_path, results_path, train_data_stats_path, train_data_ratings_distribution_path
from preprocessing.cache import load_from_pickle
from utils.helpers import get_user_data, analyze_data, generate_index_map, load_training_data, load_statistical_data

if __name__ == '__main__' :
    """
    The main function to preprocess the data
    """
    print('----------------- Analyzing Train Data -----------------')
    # Load the train data
    user_ratings = get_user_data(train_data_path)
    # Analyze the train data
    analyze_data(user_ratings, train_data_stats_path, train_data_ratings_distribution_path)
    print('--------------------------------------------------------')
    print('----------------- Preprocessing Train Data -----------------')
    # Create the index map for the items
    print("Creating the index map for the items...")
    generate_index_map(train_data_path)
    # Load the index map
    item_idx = load_from_pickle('data/cache/item_idx.pkl')
    # Get the item and user data
    print("Getting the item and user data...")
    load_training_data(train_data_path, item_idx)
    user_ratings = load_from_pickle('data/cache/user_ratings.pkl')
    item_raters = load_from_pickle('data/cache/item_raters.pkl')
    
    load_statistical_data(user_ratings, item_raters)


    stat_data = load_from_pickle('data/cache/train_statistics.pkl')
    μ = stat_data['μ']
    b_x = stat_data['b_x']
    b_i = stat_data['b_i']

    print('--------------------------------------------------------')

