from utils.config import np, pd, coo_matrix, csr_matrix, svds, EM
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

    def fit(self, user_ratings):
        """
        Train the SVD model on the given data.
        :param user_ratings: Dictionary with user as key and a list of (item, rating) tuples as value
        """
        # Transform the user_ratings data into lists for constructing the sparse matrix
        users = []
        items = []
        ratings = []

        for user, ratings_list in user_ratings.items():
            for item, rating in ratings_list:
                users.append(user)
                items.append(item)
                ratings.append(rating)

        # Create the sparse matrix using COO format
        num_users = max(users) + 1
        num_items = max(items) + 1
        coo = coo_matrix((ratings, (users, items)), shape=(num_users, num_items))
        sparse_matrix = coo.tocsr()  # Convert to CSR format

        # Load the statistics data
        stats_data = load_from_pickle('data/cache/pkls/train_statistics.pkl')
        self.μ = stats_data['μ']
        self.b_x = stats_data['b_x']
        self.b_i = stats_data['b_i']

        # If μ, b_x, or b_i are not provided, initialize them
        if self.μ is None:
            self.μ = np.mean(ratings)
        if self.b_x is None:
            self.b_x = np.zeros(num_users)
        if self.b_i is None:
            self.b_i = np.zeros(num_items)

        # Initialize latent factors using SVD
        self.P, self.Q = self.extract_factors(sparse_matrix, method="basis")
        
        # Train the model using SGD
        self.sgd(users, items, ratings)
    
    def extract_factors(self,sparse_matrix, method = "basis"):
        """
        The function to extract the latent factors from the model using SVD.
        """
        if method == 'basis':
            u, s, vt = svds(sparse_matrix, k=self.num_factors)
            P = u @ np.diag(s)
            Q = vt.T
        elif method == 'IncSVD':
            M = EM.EvolvingMatrix(sparse_matrix, k=self.num_factors, sparse=True, method="ZhaSimon")
            Uk, Sigmak, Vk = M.Uk, M.Sigmak, M.Vk
            P = Uk @ np.diag(Sigmak)
            Q = Vk
        return P, Q

    def sgd(self, users, items, ratings):
        """
        Perform stochastic gradient descent to optimize the biases and latent factors.
        """
        for epoch in range(self.num_epochs):
            for user, item, rating in zip(users, items, ratings):
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
        if user_id in range(len(self.b_x)) and item_id in range(len(self.b_i)):
            return self.predict_single(user_id, item_id)
        else:
            return self.μ  # If user or item is unknown, return the global mean

        


if __name__ == "__main__":

    user_ratings = load_from_pickle('data/cache/pkls/user_ratings.pkl')
    # Initialize and train the SVD model
    svd = SVD(num_factors=20, learning_rate=0.005, reg_bias=0.02, reg_pq=0.02, epochs=20)
    svd.fit(user_ratings)
    
    # Example prediction
    user_id = 0
    item_id = 378216
    rating = svd.predict(user_id, item_id)
    print(f'Predicted rating for user {user_id} and item {item_id}: {rating}')