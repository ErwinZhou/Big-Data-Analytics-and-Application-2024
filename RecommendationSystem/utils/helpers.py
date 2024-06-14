from preprocessing.cache import save_to_pickle, load_from_pickle
from utils.config import np, pd, defaultdict, plt, colors, math

def log_normalization(rating):
    return np.log1p(rating)  # log(1 + rating)

def min_max_normalization(rating, min_rating, max_rating, new_min=1, new_max=10):
    return (rating - min_rating) / (max_rating - min_rating) * (new_max - new_min) + new_min

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
                item_rating = f.readline().strip().split()
                item = int(item_rating[0])
                try:
                    rating = int(item_rating[1])
                except ValueError:
                    rating = None
                user_ratings[user_id].append([item, rating])
    return user_ratings

def analyze_training_data(user_ratings, train_data_stats_path, train_data_ratings_distribution_path):
    """
    This function serves to analyze the train.txt data
    It focuses on the following aspects:
    - [Users] Number of users, max user id, min user id
    - [Items] Number of rated items, max item id, min item id
    - [Ratings] Number of ratings, max rating, min rating, average rating, missing values, variance, sparseness ratio
    - [Rating Distribution] Distribution of ratings
    - [Others] (1) Is there some users never rated any items? 
    Args:
        user_ratings: dict, the user ratings
        train_data_stats_path: str, the path of the statistics file
        train_data_ratings_distribution_path: str, the path of the ratings distribution png file
    Returns:
        True or False
    """
    all_ratings = [rating for ratings in user_ratings.values() for _, rating in ratings if rating is not None]
    all_items = [item for ratings in user_ratings.values() for item, _ in ratings if item is not None]
    
    num_users = len(user_ratings)
    num_items = len(set(all_items))
    total_ratings = len(all_ratings)
    max_rating = max(all_ratings) if all_ratings else None
    min_rating = min(all_ratings) if all_ratings else None
    avg_rating = sum(all_ratings) / total_ratings if all_ratings else None
    
    unique_ratings = len(set(all_ratings))  # Calculate the number of unique rating values
    
    missing_values = sum(1 for rating in all_ratings if rating is None)

    max_item_id = max(all_items)
    min_item_id = min(all_items)
    
    max_user_id = max(user_ratings.keys())
    min_user_id = min(user_ratings.keys())

    rating_series = pd.Series(all_ratings)
    rating_distribution = rating_series.value_counts().sort_index()
    variance = rating_series.var() if all_ratings else None
    sparseness_ratio = 1 - (total_ratings / (max_user_id * max_item_id))

    # Is there some users never rated any items?
    users_never_rated = [user_id for user_id, ratings in user_ratings.items() if not ratings]

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
            print(f'  Variance: {variance:.2f}', file=f)
            print(f'  Missing values (ratings of None): {missing_values}', file=f)
            print(f'  Sparseness ratio: {sparseness_ratio:.2%}', file=f)
            print('\nRating distribution:', file=f)
            print('\n'.join(f'  {line}' for line in str(rating_distribution).split('\n')), file=f)
            print('\nOthers: ', file=f)
            print(f'  Users never rated any items: {users_never_rated}', file=f)
        
        # Plot the rating distribution with bins
        plt.hist(rating_series, bins=10, edgecolor='black', color='grey', alpha=0.5) if all_ratings else None
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Rating Distribution')
        plt.xticks(range(0, 101, 10))
        
        # Calculate the average rating and add a vertical line
        average_rating = np.mean(rating_series) if all_ratings else None
        plt.axvline(x=average_rating, color=colors[0], linestyle='--', label=f'Average Rating: {average_rating:.2f}') if average_rating is not None else None
        
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
    - [Users] : Max(Min) user id, Number of users, if there are any new users to avoid cold start
    - [Items] : Max(Min) item id, Number of items needed to be rated, if there are any new items to avoid cold start
    Finally, save the statistics to a text file 
    Args:
        test_data: dict, the test data
        test_data_stats_path: str, the path of the statistics file
        user_ratings: dict, the user ratings
        item_raters: dict, the item raters
    Returns:
        True or False
    """
    # Get the number of users and items in the test data
    num_users = len(test_data)
    num_items = len(set(item for items in test_data.values() for item in items))
    
    # Get the number of new users and items in the test data
    new_users = len(set(test_data.keys()) - set(user_ratings.keys()))
    new_items = len(set(item for items in test_data.values() for item in items if item not in item_raters))

    # Max(Min) user id and item id in the test data
    max_user_id = max(test_data.keys())
    min_user_id = min(test_data.keys())
    max_item_id = max(item for items in test_data.values() for item in items)
    min_item_id = min(item for items in test_data.values() for item in items)
    
    try:
        # Save the statistics to a text file
        with open(test_data_stats_path, 'w') as file:
            print(f'Users: ', file=file)
            print(f'   Number of users: {num_users}', file=file)
            print(f'   Max user id: {max_user_id}', file=file)
            print(f'   Min user id: {min_user_id}', file=file)
            print(f'   Number of new users: {new_users}', file=file)
            print('\nItems: ', file=file)
            print(f'  Number of items: {num_items}', file=file)
            print(f'  Max item id: {max_item_id}', file=file)
            print(f'  Min item id: {min_item_id}', file=file)
            print(f'  Number of new items: {new_items}', file=file)
    except IOError:
        return False
    return True

def get_attribute_data(attribute_path):
    """
    Load the attribute data from the file
    Args:
        attribute_path: str, the path of the attribute file
    Returns:
        attribute_data: dict, the attribute data
    """
    attribute_data = defaultdict(dict)
    with open(attribute_path, 'r') as file:
        while (line := file.readline().strip()):
            item_id, *attributes = line.split('|')
            item_id = int(item_id)
            attribute_data[item_id] = {}
            for i, attribute in enumerate(attributes, start=1):
                try:
                    attribute_data[item_id][f'attribute{i}'] = int(attribute)
                except ValueError:
                    attribute_data[item_id][f'attribute{i}'] = None
    return attribute_data
            
def analyze_attribute_data(attribute_data, attribute_stats_path, attribute_distribution_path):
    """
    This function serves to analyze the itemAttribute.txt data
    We already know that there are two types of attributes in it
    It includes the following aspects:
    - [Attributes] Max and min values of the attributes, Number of None values, ratio of None values, average value of the attributes, variance
    - [Attribute Distribution] Distribution of the attributes
    - [Others] (1)If the itemAttribute.txt file includes all the item 
               (2)The range of items in the itemAttribute.txt file
               (3)If the items are continuous
    Args:
        attribute_data: dict, the attribute data
        attribute_stats_path: str, the path of the statistics file
        attribute_distribution_path: str, the path of the attribute distribution png file
    Returns:
        True or False
    """
    attribute1_values = [data['attribute1'] for data in attribute_data.values() if data['attribute1'] is not None]
    attribute2_values = [data['attribute2'] for data in attribute_data.values() if data['attribute2'] is not None]
    total_values = len(attribute_data) * 2
    max_attribute1 = max(attribute1_values) if attribute1_values else None
    min_attribute1 = min(attribute1_values) if attribute1_values else None
    max_attribute2 = max(attribute2_values) if attribute2_values else None
    min_attribute2 = min(attribute2_values) if attribute2_values else None
    
    num_none_values1 = sum(1 for value in attribute_data.values() if value['attribute1'] is None)
    num_none_values2 = sum(1 for value in attribute_data.values() if value['attribute2'] is None)

    total_none_values = num_none_values1 + num_none_values2

    none_ratio1 = num_none_values1 / len(attribute_data) * 100
    none_ratio2 = num_none_values2 / len(attribute_data) * 100
    total_none_ratio = total_none_values / total_values * 100
    
    avg_attribute1 = sum(attribute1_values) / len(attribute1_values) if attribute1_values else None
    avg_attribute2 = sum(attribute2_values) / len(attribute2_values) if attribute2_values else None
    var_attribute1 = np.var(attribute1_values) if attribute1_values else None
    var_attribute2 = np.var(attribute2_values) if attribute2_values else None

    # (2)The range of items in the itemAttribute.txt file
    max_item_id = max(attribute_data.keys())
    min_item_id = min(attribute_data.keys())
    # (3)If the items are continuous
    continuous = False
    if len(attribute_data) == max_item_id - min_item_id + 1:
        continuous = True

    try:
        with open(attribute_stats_path, 'w') as file:
            print(f'Attribute 1: ', file=file)
            print(f'  Max value: {max_attribute1}', file=file)
            print(f'  Min value: {min_attribute1}', file=file)
            print(f'  Average value: {avg_attribute1:.2f}', file=file) if avg_attribute1 is not None else print(f'  Average value: None', file=file)
            print(f'  Variance: {var_attribute1:.2f}', file=file) if var_attribute1 is not None else print(f'  Variance: None', file=file)
            print(f'  Number of None values: {num_none_values1}', file=file)
            print(f'  Ratio of None values: {none_ratio1:.2f}%', file=file)  
            print('\nAttribute 2: ', file=file)
            print(f'  Max value: {max_attribute2}', file=file)
            print(f'  Min value: {min_attribute2}', file=file)
            print(f'  Average value: {avg_attribute2:.2f}', file=file) if avg_attribute2 is not None else print(f'  Average value: None', file=file)
            print(f'  Variance: {var_attribute2:.2f}', file=file) if var_attribute2 is not None else print(f'  Variance: None', file=file)
            print(f'  Number of None values: {num_none_values2}', file=file)
            print(f'  Ratio of None values: {none_ratio2:.2f}%', file=file)  
            print('\n', file=file)
            print(f'Total number of None values: {total_none_values}', file=file)
            print(f'Total ratio of None values: {total_none_ratio:.2f}%', file=file)  
            print('\nOthers: ', file=file)
            print(f'  The range of items in the itemAttribute.txt file: {min_item_id} - {max_item_id}', file=file)
            print(f'  If the items are continuous: {continuous}', file=file)

        bins = np.linspace(min(min(attribute1_values), min(attribute2_values)), 
                           max(max(attribute1_values), max(attribute2_values)), 11)

        width = (bins[1] - bins[0]) / 2

        plt.figure(figsize=(10, 6))
        if attribute1_values:
            plt.hist(attribute1_values, bins=bins, edgecolor='black', color=colors[3], alpha=0.5, label='Attribute 1', align='mid')
            plt.axvline(avg_attribute1, color=colors[3], linestyle='dashed', linewidth=2, label='Attribute 1 Mean')
        if attribute2_values:
            plt.hist(attribute2_values, bins=bins, edgecolor='black', color=colors[0], alpha=0.5, label='Attribute 2', align='mid')
            plt.axvline(avg_attribute2, color=colors[0], linestyle='dashed', linewidth=2, label='Attribute 2 Mean')
        
        plt.xlabel('Attribute Value')
        plt.ylabel('Count')
        plt.title('Attribute Distribution')
        
        labels = [f'{math.ceil((bins[i]+bins[i+1])/2):.1f}' for i in range(len(bins)-1)]
        plt.xticks(bins[:-1], labels=labels, rotation=45)

        plt.legend(loc="lower left", frameon=True)
        plt.tight_layout()


        plt.savefig(attribute_distribution_path)
        plt.show()

    except IOError:
        return False
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
    
    log_ratings = []
    # First pass to apply log transformation and collect all log ratings
    with open(train_data_path, 'r') as file:
        while (line := file.readline().strip()):
            user_id, num_ratings = map(int, line.split('|'))
            for _ in range(num_ratings):
                line = file.readline().strip()
                item_id, rating = map(int, line.split())
                log_rating = log_normalization(rating)
                log_ratings.append(log_rating)

    min_log_rating = np.min(log_ratings)
    max_log_rating = np.max(log_ratings)
    
    # Second pass to normalize log ratings to the desired range
    with open(train_data_path, 'r') as file:
        while (line := file.readline().strip()):
            user_id, num_ratings = map(int, line.split('|'))
            for _ in range(num_ratings):
                line = file.readline().strip()
                item_id, rating = map(int, line.split())
                log_rating = log_normalization(rating)
                normalized_rating = min_max_normalization(log_rating, min_log_rating, max_log_rating, 0, 10)
                user_ratings[user_id].append([index_map[item_id], normalized_rating])
                item_raters[index_map[item_id]].append([user_id, normalized_rating])
    
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

def split_training_data(user_ratings, validation_ratio=0.2, shuffle=True, random_seed=42):
    """
    Split the training data into training and validation sets
    Because the required result form is listed as [User]{Item1: Rating}{Item2: Rating}...
    So only to shuffle the user_ratings and split the data into training and validation sets(item_raters is not needed here)
    Args:
        user_ratings: dict, the user ratings
        validation_ratio: float, the ratio of the validation set
        shuffle: bool, whether to shuffle the data
        random_seed: int, the random seed
    Returns:
        train_data: dict, the training data
        validation_data: dict, the validation data
    """
    if shuffle:
        np.random.seed(random_seed)
        for user_id, ratings in user_ratings.items():
            np.random.shuffle(ratings)
    train_data, validation_data = {}, {}
    for user_id, ratings in user_ratings.items():
        split_index = int(len(ratings) * (1 - validation_ratio))
        train_data[user_id] = ratings[:split_index]
        validation_data[user_id] = ratings[split_index:]
    return train_data, validation_data

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