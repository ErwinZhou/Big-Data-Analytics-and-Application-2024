from preprocessing.cache import save_to_pickle, load_from_pickle
from utils.config import np, pd, defaultdict, plt, colors


def get_user_data(file_path):
    """
    Load the data from the file
    Only using to load the userr data before creating the index
    Args:
        file_path: str, the path of the data file
    Returns:
        user_ratings: dict, the user ratings
    """
    user_ratings = defaultdict(list)
    with open(file_path, 'r') as f:
        while (line := f.readline().strip()):
            user_id, num_ratings = map(int, line.split('|'))
            for _ in range(num_ratings):
                item, rating = map(int, f.readline().strip().split())
                user_ratings[user_id].append([item, rating])
    return user_ratings

def analyze_training_data(user_ratings, train_data_stats_path, train_data_ratings_distribution_path):
    """
    This function serves to analyze the train.txt data
    It focuses on the following aspects:
    - [Users] Number of users, max user id, min user id
    - [Items] Number of rated items, max item id, min item id
    - [Ratings] Number of ratings, max rating, min rating, average rating, missing values
    - [Rating Distribution] Distribution of ratings
    Args:
        user_ratings: dict, the user ratings
        train_data_stats_path: str, the path of the statistics file
        train_data_ratings_distribution_path: str, the path of the ratings distribution png file
    Returns:
        True or False
    """
    all_ratings = [rating for ratings in user_ratings.values() for _, rating in ratings]
    all_items = [item for ratings in user_ratings.values() for item, _ in ratings]
    
    num_users = len(user_ratings)
    num_items = len(set(all_items))
    total_ratings = len(all_ratings)
    
    unique_ratings = len(set(all_ratings))  # Calculate the number of unique rating values
    
    max_rating = max(all_ratings)
    min_rating = min(all_ratings)
    avg_rating = sum(all_ratings) / total_ratings
    missing_values = sum(1 for rating in all_ratings if rating is None)

    max_item_id = max(all_items)
    min_item_id = min(all_items)
    
    max_user_id = max(user_ratings.keys())
    min_user_id = min(user_ratings.keys())

    rating_series = pd.Series(all_ratings)
    rating_distribution = rating_series.value_counts().sort_index()

    sparseness_ratio = 1 - (total_ratings / (max_user_id * max_item_id))


    # Save statistics to a text file
    try:
        with open(train_data_stats_path, 'w') as f:
            print(f'Users: ', file=f)
            print(f'   Number of users: {num_users}', file=f)
            print(f'   Max user id: {max_user_id}', file=f)
            print(f'   Min user id: {min_user_id}', file=f)
            print('\nItems: ', file=f)
            print(f'  Number of rated items: {num_items}', file=f)
            print(f'  Max item id: {max_item_id}', file=f)
            print(f'  Min item id: {min_item_id}', file=f)
            print('\nRatings: ', file=f)
            print(f'  Number of ratings: {total_ratings}', file=f)
            print(f'  Number of unique rating values: {unique_ratings}', file=f)  # Add the number of unique rating values
            print(f'  Max rating: {max_rating}', file=f)
            print(f'  Min rating: {min_rating}', file=f)
            print(f'  Average rating: {avg_rating:.2f}', file=f)
            print(f'  Missing values (ratings of None): {missing_values}', file=f)
            print(f'  Sparseness ratio: {sparseness_ratio:.2%}', file=f)
            print('\nRating distribution:', file=f)
            print('\n'.join(f'  {line}' for line in str(rating_distribution).split('\n')), file=f)
        
        # Plot the rating distribution with bins
        plt.hist(rating_series, bins=10, edgecolor='black', color='grey', alpha=0.5)
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Rating Distribution')
        plt.xticks(range(0, 101, 10))
        
        # Calculate the average rating and add a vertical line
        average_rating = np.mean(rating_series)
        plt.axvline(x=average_rating, color=colors[0], linestyle='--', label=f'Average Rating: {average_rating:.2f}')
        
        # Calculate the average count and add a horizontal line
        average_count = np.mean(rating_distribution)
        plt.axhline(y=average_count, color=colors[3], linestyle='--', label=f'Average Count: {average_count:.2f}')
        
        plt.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        
        # Save the plot to a PNG file
        plt.savefig(train_data_ratings_distribution_path)
        
        plt.show()

        return True
    except IOError:
        return False

def get_test_data(test_data_path):
    """
    Load the test data from the file
    Args:
        test_data_path: str, the path of the test data file
    Returns:
        test_data: dict, the test data
    """
    test_data = defaultdict(list)
    with open(test_data_path, 'r') as file:
        while (line := file.readline().strip()):
            user_id, num_ratings = map(int, line.split('|'))
            for _ in range(num_ratings):
                item_id = int(file.readline().strip())
                test_data[user_id].append(item_id)
    return test_data

def analyze_test_data(test_data, test_data_stats_path, user_ratings, item_raters):
    """
    This function serves to analyze the test.txt data
    To find out the potential setbacks and missing values:
    - [Users] : Number of users, if there are any new users to avoid cold start
    - [Items] : Number of items needed to be rated, if there are any new items to avoid cold start
    """
    # Get the number of users and items in the test data
    num_users = len(test_data)
    num_items = len(set(item for items in test_data.values() for item in items))
    
    # Get the number of new users and items in the test data
    new_users = len(set(test_data.keys()) - set(user_ratings.keys()))
    new_items = len(set(item for items in test_data.values() for item in items if item not in item_raters))
    
    # Save the statistics to a text file
    with open(test_data_stats_path, 'w') as file:
        print(f'Users: ', file=file)
        print(f'   Number of users: {num_users}', file=file)
        print(f'   Number of new users: {new_users}', file=file)
        print('\nItems: ', file=file)
        print(f'  Number of items: {num_items}', file=file)
        print(f'  Number of new items: {new_items}', file=file)

    return True

def generate_index_map(train_data_path):
    """
    Because the sequence of the items is not continuous
    A index map is needed to map the item id to a continuous index
    Finally, the index map will be dumped to item_idx.txt and item_idx.pkl
    Args:
        train_data_path: str, the path of the data file
    Returns:
        True or False
    """
    unique_items = set()
    with open(train_data_path, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) == 2:
                _, count = map(int, parts)
                for _ in range(count):
                    line = next(file)
                    item, _ = map(int, line.strip().split())
                    unique_items.add(item)
    sorted_items = sorted(list(unique_items))
    index_map = {value: index for index, value in enumerate(sorted_items)}
    # Save the index map to a txt file
    with open('data/cache/txts/item_idx.txt', 'w') as file:
        for item_id, index in index_map.items():
            print(f'{item_id} {index}', file=file)
    # Dump the index map to index pickle file
    if save_to_pickle(index_map, 'data/cache/pkls/item_idx.pkl'):
        return True
    return False

def load_training_data(train_data_path, index_map):
    """
    Load the training data to store in the user_data and item_data dictionaries
    After creating the index, update user_ratings and item_raters
    Finally,  user_ratings and item_raters will be dumped to user_ratings.pkl and item_raters.pkl
    Args:
        train_data_path: str, the path of the training data
        index_map: dict, the index map of the items, for there is no continuous index for the items
    Returns:
        True or False 
    """
    user_ratings, item_raters = defaultdict(list), defaultdict(list)
    with open(train_data_path, 'r') as file:
        while (line := file.readline().strip()):
            user_id, num_ratings = map(int, line.split('|'))
            for _ in range(num_ratings):
                line = file.readline().strip()
                item_id, rating = map(int, line.split())
                # normalized_rating = rating / 10.0 
                # user_data[user_id].append([index_map[item_id], normalized_rating])
                # item_data[index_map[item_id]].append([user_id, normalized_rating])
                user_ratings[user_id].append([index_map[item_id], rating]) 
                item_raters[index_map[item_id]].append([user_id, rating]) # Only if the item is previously stored in the index map and rated by the users
    
    # Save the user_ratings and item_raters to txt files
    with open('data/cache/txts/user_ratings.txt', 'w') as file:
        for user_id, ratings in user_ratings.items():
            print(f'{user_id} {len(ratings)}', file=file)
            for item_id, rating in ratings:
                print(f'{item_id} {rating}', file=file)

    with open('data/cache/txts/item_raters.txt', 'w') as file:
        for item_id, raters in item_raters.items():
            print(f'{item_id} {len(raters)}', file=file)
            for user_id, rating in raters:
                print(f'{user_id} {rating}', file=file)
    

    # Save the user_ratings and item_raters to pikle files
    if save_to_pickle(user_ratings, 'data/cache/pkls/user_ratings.pkl') and save_to_pickle(item_raters, 'data/cache/pkls/item_raters.pkl'):
        return True
    return False

def load_attribute_data(attribute_path, index_map):
    """

    """

def load_statistical_data(user_ratings, item_raters):
    """
    This function serves to calculate the statistical features of the training data.
    It includes the following aspects:
    - μ(mu): The global average rating of all users
    - b_x: The average rating deviation of user x
    - b_i: The average rating deviation of item i
    Finally, save the statistical features to train_statistics.pkl
    Args:
        user_ratings: {User ID: [[Item ID, Rating]]}
        item_raters: {Item ID: [[User ID, Rating]]}

    Returns:
        True or False
    """
    # Initialize the statistical features
    μ = 0.0
    total_ratings = 0
    b_x = np.zeros(len(user_ratings), dtype=np.float64)
    b_i = np.zeros(len(item_raters), dtype=np.float64)

    # Calculate global average rating (μ) and user bias (b_x)
    for user_id, ratings in user_ratings.items():
        user_total_rating = sum(rating for _, rating in ratings)
        μ += user_total_rating
        total_ratings += len(ratings)
        b_x[user_id] = user_total_rating / len(ratings)

    μ /= total_ratings

    # Calculate item bias (b_i)
    for item_id, ratings in item_raters.items():
        item_total_rating = sum(rating for _, rating in ratings)
        b_i[item_id] = item_total_rating / len(ratings)

    # Calculate deviations from the global average
    b_x -= μ
    b_i -= μ

    if save_to_pickle({'μ': μ, 'b_x': b_x, 'b_i': b_i}, 'data/cache/pkls/train_statistics.pkl'):
        return True
    return False