from utils.config import np, pd, csr_matrix, svds
from utils.evaluation import SSE, RMSE
from preprocessing.cache import load_from_pickle, save_to_pickle

class SVD:
    """
    The class to implement the Singular Value Decomposition algorithm
    """
    def __init__(self, num_factors=20):
        self.num_factors = num_factors
        self.μ = None
        self.b_x = None
        self.b_i = None
        self.q = None
        self.p = None

    def fit(self, train_data):
        """
        Train the SVD model on the given data.
        
        :param train_data: Pandas DataFrame with columns ['user', 'item', 'rating']
        """
        # Create user-item matrix
        user_item_matrix = train_data.pivot(index='user', columns='item', values='rating').fillna(0)
        sparse_matrix = csr_matrix(user_item_matrix)
        
        # Global mean
        self.μ = train_data['rating'].mean()

        # Calculate biases
        self.b_x = train_data.groupby('user')['rating'].mean() - self.μ
        self.b_i = train_data.groupby('item')['rating'].mean() - self.μ

        # Fill missing values with zeros for biases
        self.b_x = self.b_x.reindex(user_item_matrix.index).fillna(0)
        self.b_i = self.b_i.reindex(user_item_matrix.columns).fillna(0)

        # SVD
        u, s, vt = svds(sparse_matrix, k=self.num_factors)
        self.q = vt.T
        self.p = u @ np.diag(s)

    def predict(self, user_id, item_id):
        """
        Predict the rating of a given user for a given item.
        
        :param user_id: int, the user ID
        :param item_id: int, the item ID
        :return: float, the predicted rating
        """
        if user_id in self.b_x.index and item_id in self.b_i.index:
            b_x = self.b_x[user_id]
            b_i = self.b_i[item_id]
            q_i = self.q[self.b_i.index.get_loc(item_id)]
            p_x = self.p[self.b_x.index.get_loc(user_id)]
            return self.μ + b_x + b_i + np.dot(q_i, p_x)
        else:
            return self.μ  # If user or item is unknown, return the global mean

    def calc_bias(self):
        """
        The function to calculate the bias for the items.
        Without the factorization, the bias is calculated as:
        b_i = μ + b_x + b_i
        """
        stat_data = self.load_from_pickle('data/cache/pkls/train_statistics.pkl')
        self.μ = stat_data['μ']
        self.b_x = stat_data['b_x']
        self.b_i = stat_data['b_i']
        self.bias = self.μ + self.b_x + self.b_i


if __name__ == "__main__":
    # Load training data
    train_path = 'data/processed_train.csv'
    train_df = pd.read_csv(train_path)
    
    # Initialize and train the SVD model
    svd = SVD(num_factors=20)
    svd.fit(train_df)
    
    # Example prediction
    user_id = 0
    item_id = 378216
    rating = svd.predict(user_id, item_id)
    print(f'Predicted rating for user {user_id} and item {item_id}: {rating}')
