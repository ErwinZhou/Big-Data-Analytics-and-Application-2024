from utils.config import train_data_path, test_data_path, attributes_path, results_path, train_data_stats_path, train_data_ratings_distribution_path, test_data_stats_path, attributes_stats_path, attributes_distribution_path, processed_attributes_path
from preprocessing.cache import load_from_pickle
from utils.helpers import get_user_data, analyze_training_data, generate_index_map, load_training_data, load_statistical_data, get_test_data, analyze_test_data, get_attribute_data, analyze_attribute_data, process_item_attributes

if __name__ == '__main__' :
    """
    The main function to preprocess the data
    """
    # print('----------------- Analyzing Train Data -----------------')
    # # Load the train data
    # user_ratings = get_user_data(train_data_path)
    # # Analyze the train data
    # analyze_training_data(user_ratings, train_data_stats_path, train_data_ratings_distribution_path)
    # print("[INFO] Train data analysis completed.")
    # print('--------------------------------------------------------')
    # print('----------------- Preprocessing Train Data -----------------')
    # # Create the index map for the items
    # print("Creating the index map for the items...")
    # generate_index_map(train_data_path)
    # # Load the index map
    # item_idx = load_from_pickle('data/cache/pkls/item_idx.pkl')
    # # Get the item and user data
    # print("Getting the item and user data...")
    # load_training_data(train_data_path, item_idx)
    # user_ratings = load_from_pickle('data/cache/pkls/user_ratings.pkl')
    # item_raters = load_from_pickle('data/cache/pkls/item_raters.pkl')
    
    
    # print('[INFO] Preprocessing Train Data completed.')
    # # Analyze the test data to analyze if there is cold-start problem or not
    # print('----------------- Analyzing Test Data -----------------')
    # test_data = get_test_data(test_data_path)
    # analyze_test_data(test_data, test_data_stats_path, user_ratings, item_raters)
    # print("[INFO] Test data analysis completed.")
    # print('----------------- Analyzing Attributes Data -----------------')
    # attributes_data = get_attribute_data(attributes_path)
    # analyze_attribute_data(attributes_data, attributes_stats_path, attributes_distribution_path)
    # print("[INFO] Attributes data analysis completed.")
    # print('--------------------------------------------------------')
    # load_statistical_data(user_ratings, item_raters)
    # stat_data = load_from_pickle('data/cache/pkls/train_statistics.pkl')
    # μ = stat_data['μ']
    # b_x = stat_data['b_x']
    # b_i = stat_data['b_i']

    process_item_attributes(attributes_path, processed_attributes_path)
    # print('--------------------------------------------------------')

