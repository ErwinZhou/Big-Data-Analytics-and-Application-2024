from models.cf import CF
from utils.config import pd, np
from models.svd import SVD



if __name__ == "__main__":
    # Load training data

    
    # Initialize and train the SVD model
    svd = SVD(num_factors=100)
    
    # Example prediction
    user_id = 0
    item_id = 378216
    rating = svd.predict(user_id, item_id)
    print(f'Predicted rating for user {user_id} and item {item_id}: {rating}')