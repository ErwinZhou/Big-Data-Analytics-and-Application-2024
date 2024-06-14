from utils.config import np, pd, csr_matrix, svds
from utils.evaluation import SSE, RMSE
from preprocessing.cache import load_from_pickle, save_to_pickle

class SVD:
    """
    The class to implement the Singular Value Decomposition algorithm
    """
    def __init__(self, num_factors=20, learning_rate=0.005, reg_bias=0.02, reg_pq=0.02, epochs=20):
        self.num_factors = num_factors  # The dimension of the latent space
        self.learning_rate = learning_rate
        self.reg_bias = reg_bias # Regularization parameter for user and item biases to prevent overfitting
        self.reg_pq = reg_pq # Regularization parameter for user and item latent factors to prevent overfitting
        self.epochs = epochs
        self.μ = None  # The global mean of the ratings
        self.b_x = None # User biases [n_users] Treated as parameters to be learned
        self.b_i = None # Item biases [n_items] Treated as parameters to be learned
        self.P = None  # User matrix of the SVD [n_users, num_factors]
        self.Q = None  # Item matrix of the SVD [n_items, num_factors]

    def fit(self, train_data):
        """
        Train the SVD model on the given data.
        
        :param train_data: Pandas DataFrame with columns ['user', 'item', 'rating']
        """
        # Create user-item matrix
        user_item_matrix = train_data.pivot(index='user', columns='item', values='rating').fillna(0)
        sparse_matrix = csr_matrix(user_item_matrix)
        
        # load the statistics data
        stats_data = load_from_pickle('data/cache/pkls/train_statistics.pkl')
        self.μ = stats_data['μ']
        self.b_x = stats_data['b_x']
        self.b_i = stats_data['b_i']
        
        # If μ, b_x, or b_i are not provided, initialize them
        if self.μ is None:
            self.μ = train_data['rating'].mean()
        if self.b_x is None:
            self.b_x = np.zeros(user_item_matrix.shape[0])
        if self.b_i is None:
            self.b_i = np.zeros(user_item_matrix.shape[1])

        # Initialize latent factors using SVD
        u, s, vt = svds(sparse_matrix, k=self.num_factors)
        self.P = u @ np.diag(s)
        self.Q = vt.T

        # Train the model using SGD
        self.sgd(train_data)

    def sgd(self, train_data):
        """
        Perform stochastic gradient descent to optimize the biases and latent factors.
        """
        for _ in range(self.epochs):
            for row in train_data.itertuples():
                user = row.user
                item = row.item
                rating = row.rating

                prediction = self.predict_single(user, item)
                error = rating - prediction

                # Update biases
                self.b_x[user] += self.learning_rate * (error - self.reg_bias * self.b_x[user])
                self.b_i[item] += self.learning_rate * (error - self.reg_bias * self.b_i[item])

                # Update latent factors
                self.P[user, :] += self.learning_rate * (error * self.Q[item, :] - self.reg_pq * self.P[user, :])
                self.Q[item, :] += self.learning_rate * (error * self.P[user, :] - self.reg_pq * self.Q[item, :])

    def predict_single(self, user, item):
        """
        Predict a single rating.
        
        :param user: int, the user ID
        :param item: int, the item ID
        :return: float, the predicted rating
        """
        prediction = self.μ + self.b_x[user] + self.b_i[item] + np.dot(self.P[user, :], self.Q[item, :])
        return prediction

    def predict(self, user_id, item_id):
        """
        Predict the rating of a given user for a given item.
        
        :param user_id: int, the user ID
        :param item_id: int, the item ID
        :return: float, the predicted rating
        """
        if user_id in self.b_x.index and item_id in self.b_i.index:
            return self.predict_single(user_id, item_id)
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
        


# if __name__ == "__main__":
#     # Load training data
#     train_path = 'data/processed_train.csv'
#     train_df = pd.read_csv(train_path)
    
#     # Initialize and train the SVD model
#     svd = SVD(num_factors=20, learning_rate=0.005, reg_bias=0.02, reg_pq=0.02, epochs=20)
#     svd.fit(train_df)
    
#     # Example prediction
#     user_id = 0
#     item_id = 378216
#     rating = svd.predict(user_id, item_id)
#     print(f'Predicted rating for user {user_id} and item {item_id}: {rating}')