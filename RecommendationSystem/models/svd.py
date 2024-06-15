from utils.config import np, pd, coo_matrix, csr_matrix, svds, EM, logging, tqdm, plt, pickle
from utils.evaluation import SSE, RMSE
from preprocessing.cache import load_from_pickle, save_to_pickle
from utils.config import losses_png_path, models_path, processed_attributes_path, evaluation_path

class SVD:
    """
    The class to implement the Singular Value Decomposition algorithm
    """
    def __init__(self, num_factors=100, learning_rate=0.001, reg_bias=0.02, reg_pq=0.02, reg_weight=0.02, epochs=20, train_ratio = 0.8, shuffle = True, svd_method="basis", training_method="basis"):
        self.num_factors = num_factors  # The dimension of the latent space
        self.learning_rate = learning_rate
        self.reg_bias = reg_bias  # Regularization parameter for user and item biases to prevent overfitting
        self.reg_pq = reg_pq  # Regularization parameter for user and item latent factors to prevent overfitting
        self.reg_weight = reg_weight  # Regularization parameter for the attribute weights to prevent overfitting
        self.epochs = epochs
        self.μ = None  # The global mean of the ratings
        self.b_x = None  # User biases [n_users] Treated as parameters to be learned
        self.b_i = None  # Item biases [n_items] Treated as parameters to be learned
        self.P = None  # User matrix of the SVD [n_users, num_factors]
        self.Q = None  # Item matrix of the SVD [n_items, num_factors]
        self.shuffle = shuffle  # Whether to shuffle the ratings of each user
        self.train_ratio = train_ratio  # The ratio of the training set
        self.svd_method = svd_method  # The method to extract the latent factors, either 'basis' or 'IncSVD'
        self.training_method = training_method  # The method to train the model, 'basis' or 'SVD++' or 'AttributesSVD++'
        self.losses = []  # List to store the RMSE for each epoch
        self.item_attribute1 = None
        self.item_attribute2 = None
        self.attribute_weight = None

        self.train_set = None
        self.eval_set = None

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)

    def fit(self, user_ratings):
        """
        Train the SVD model on the given data.
        :param user_ratings: Dictionary with user as key and a list of (item, rating) tuples as value
        """
        self.logger.info("Starting training")
        # Transform the user_ratings data into lists for constructing the sparse matrix
        users = []
        items = []
        ratings = []
        
        for user, ratings_list in self.train_set.items():
            for item, rating in ratings_list:
                users.append(user)
                items.append(item)
                ratings.append(float(rating))

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

        if self.training_method == 'SVD++' or self.training_method == 'AttributesSVD++':
            self.P, self.Q = self.extract_factors(sparse_matrix) # Initialize latent factors using SVD
        if self.training_method == 'AttributesSVD++':
            self.item_attribute1, self.item_attribute2 = self.load_item_attributes(processed_attributes_path)
            self.attribute_weight = [0.5, 0.5]
        
        # Train the model using SGD
        self.sgd(users, items, ratings)
        self.plot_losses()

    def extract_factors(self, sparse_matrix):
        """
        The function to extract the latent factors from the model using SVD.
        """
        if self.svd_method == 'basis':
            u, s, vt = svds(sparse_matrix, k=self.num_factors)
            P = u @ np.diag(s)
            Q = vt.T
        elif self.svd_method == 'IncSVD':
            M = EM.EvolvingMatrix(sparse_matrix, k=self.num_factors, sparse=True, method="ZhaSimon")
            Uk, Sigmak, Vk = M.Uk, M.Sigmak, M.Vk
            P = Uk @ np.diag(Sigmak)
            Q = Vk
        return P, Q
    
    def split_data(self, user_ratings):
        """
        Split the data into training and evaluation sets.
        Shuffle the ratings of each user and split them into self.train_ratio and 1 - self.train_ratio.
        """
        train_set = {}
        eval_set = {}
        for user, ratings in user_ratings.items():
            if self.shuffle:
                np.random.shuffle(ratings)
            split_idx = int(len(ratings) * self.train_ratio)
            train_set[user] = ratings[:split_idx]
            eval_set[user] = ratings[split_idx:]
        
        # Split the data into training and evaluation sets
        self.train_set, self.eval_set = train_set, eval_set
        
    

    def evaluate(self):
        """
        Evaluate the model on the given data.
        """
        self.logger.info("Starting evaluation")
        errors = []
        for user, ratings_list in self.eval_set.items():
            for item, true_rating in ratings_list:
                predicted_rating = self.predict(user, item)
                error = true_rating - predicted_rating
                errors.append(error)
        rmse = np.sqrt(np.mean(np.square(errors)))
        self.logger.info(f"Evaluation completed. RMSE: {rmse:.4f}")

        current_evaluation_path = evaluation_path + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(self.num_factors, self.learning_rate, self.reg_bias, self.reg_pq, self.reg_weight, self.epochs, self.train_ratio, self.shuffle, self.svd_method, self.training_method, rmse)
        # Save the evaluation results to a txt and pickle file
        with open(current_evaluation_path, 'w') as f:
            f.write(f"RMSE: {rmse:.4f}")
        
        current_evaluation_path.replace('txts', 'pkls')
        current_evaluation_path.replace('.txt', '.pkl')
        save_to_pickle(errors, current_evaluation_path)
        return rmse
        


        

    def sgd(self, users, items, ratings):
        """
        Perform stochastic gradient descent to optimize the biases and latent factors.
        """
        for epoch in range(self.epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            epoch_errors = []

            for user, item, rating in tqdm(zip(users, items, ratings), total=len(ratings), desc=f"Epoch {epoch + 1}/{self.epochs}"):
                prediction = self.predict_single(user, item)
                error = rating - prediction
                epoch_errors.append(error)

                if self.training_method == 'basis':
                    # Update biases
                    b_x_grad = self.learning_rate * (error - self.reg_bias * self.b_x[user])
                    b_i_grad = self.learning_rate * (error - self.reg_bias * self.b_i[item])

                    # Clip gradients to avoid exploding gradients
                    b_x_grad = np.clip(b_x_grad, -1.0, 1.0)
                    b_i_grad = np.clip(b_i_grad, -1.0, 1.0)

                    # Apply gradients
                    self.b_x[user] += b_x_grad
                    self.b_i[item] += b_i_grad
                
                if self.training_method == 'SVD++':
                    # Update biases
                    b_x_grad = self.learning_rate * (error - self.reg_bias * self.b_x[user])
                    b_i_grad = self.learning_rate * (error - self.reg_bias * self.b_i[item])

                    # Update latent factors
                    P_grad = self.learning_rate * (error * self.Q[item, :] - self.reg_pq * self.P[user, :])
                    Q_grad = self.learning_rate * (error * self.P[user, :] - self.reg_pq * self.Q[item, :])

                    # Clip gradients to avoid exploding gradients
                    b_x_grad = np.clip(b_x_grad, -1.0, 1.0)
                    b_i_grad = np.clip(b_i_grad, -1.0, 1.0)
                    P_grad = np.clip(P_grad, -1.0, 1.0)
                    Q_grad = np.clip(Q_grad, -1.0, 1.0)

                    # Apply gradients
                    self.b_x[user] += b_x_grad
                    self.b_i[item] += b_i_grad
                    self.P[user, :] += P_grad
                    self.Q[item, :] += Q_grad


                # Update attribute weights
                if self.training_method == 'AttributesSVD++':
                    # Update biases
                    b_x_grad = self.learning_rate * (error - self.reg_bias * self.b_x[user])
                    b_i_grad = self.learning_rate * (error - self.reg_bias * self.b_i[item])

                    # Update latent factors
                    P_grad = self.learning_rate * (error * self.Q[item, :] - self.reg_pq * self.P[user, :])
                    Q_grad = self.learning_rate * (error * self.P[user, :] - self.reg_pq * self.Q[item, :])

                    # Update the attribute weights
                    attribute1_grad = self.learning_rate * (error * self.item_attribute1[item] - self.reg_weight * self.attribute_weight[0])
                    attribute2_grad = self.learning_rate * (error * self.item_attribute2[item] - self.reg_weight * self.attribute_weight[1])

                    # Clip gradients to avoid exploding gradients
                    b_x_grad = np.clip(b_x_grad, -1.0, 1.0)
                    b_i_grad = np.clip(b_i_grad, -1.0, 1.0)
                    P_grad = np.clip(P_grad, -1.0, 1.0)
                    Q_grad = np.clip(Q_grad, -1.0, 1.0)
                    attribute1_grad = np.clip(attribute1_grad, -1.0, 1.0)
                    attribute2_grad = np.clip(attribute2_grad, -1.0, 1.0)

                    # Apply gradients
                    self.b_x[user] += b_x_grad
                    self.b_i[item] += b_i_grad
                    self.P[user, :] += P_grad
                    self.Q[item, :] += Q_grad
                    self.attribute_weight[0] += attribute1_grad
                    self.attribute_weight[1] += attribute2_grad
                
            rmse = np.sqrt(np.mean(np.square(epoch_errors)))
            self.losses.append(rmse)
            self.logger.info(f"Epoch {epoch + 1} completed. RMSE: {rmse:.4f}")

    def plot_losses(self):
        """
        Plot the RMSE loss curve over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.losses, marker='o', linestyle='-', color='b')
        plt.title('RMSE over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.savefig(f'{losses_png_path}_{self.num_factors}_{self.learning_rate}_{self.reg_bias}_{self.reg_pq}_{self.reg_weight}_{self.epochs}_{self.svd_method}_{self.training_method}.png')
        plt.show()
       
    def predict_single(self, user, item):
        """
        Predict a single rating.
        
        :param user: int, the user ID
        :param item: int, the item ID
        :return: float, the predicted rating
        """
        # Basis prediction
        
        if self.training_method == 'basis' or self.training_method == 'SVD++' or self.training_method == 'AttributesSVD++':
            prediction = self.μ + self.b_x[user] + self.b_i[item]
        if self.training_method == 'SVD++' or self.training_method == 'AttributesSVD++':
            prediction += np.dot(self.P[user, :], self.Q[item, :]) # SVD++ prediction
        if self.training_method == 'AttributesSVD++':   
            prediction += self.attribute_weight[0] * self.item_attribute1[item] + self.attribute_weight[1] * self.item_attribute2[item] # Attributes and SVD++ prediction

        if prediction > 10:
            prediction = 10
        elif prediction < 0:
            prediction = 0
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
    
    def load_item_attributes(self, processed_data_path):
        item_attribute1 = []
        item_attribute2 = []

        with open(processed_data_path, 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                item_id = int(parts[0])
                attribute_1 = float(parts[1])
                attribute_2 = float(parts[2])
                item_attribute1.append(attribute_1)
                item_attribute2.append(attribute_2)
        
        return item_attribute1, item_attribute2

    def save_model(self, path):
        """
        Save the trained model to a file.
        :param path: str, the file path to save the model
        """
        model_data = {
            'num_factors': self.num_factors,
            'learning_rate': self.learning_rate,
            'reg_bias': self.reg_bias,
            'reg_pq': self.reg_pq,
            'reg_weight': self.reg_weight,
            'epochs': self.epochs,
            'μ': self.μ,
            'b_x': self.b_x,
            'b_i': self.b_i,
            'P': self.P,
            'Q': self.Q,
            'attribute_weight': self.attribute_weight,
        }

        # print(model_data)

        # Save to pickle file
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        # Save to txt file
        path = path.replace('.pkl', '.txt')
        path = path.replace('pkls', 'txts')

        with open(path, 'w') as f:
            f.write(f"Model: SVD\n")
            f.write(f"Number of factors: {self.num_factors}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"Regularization parameter for bias: {self.reg_bias}\n")
            f.write(f"Regularization parameter for latent factors: {self.reg_pq}\n")
            f.write(f"Regularization parameter for attribute weights: {self.reg_weight}\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Global mean: {self.μ}\n")
            f.write(f"User biases: {self.b_x}\n")
            f.write(f"Item biases: {self.b_i}\n")
            f.write(f"Latent Matrix P: {self.P}\n")
            f.write(f"Latent Matrix Q: {self.Q}\n")
            f.write(f"Attribute weight: {self.attribute_weight}\n")
        
        self.logger.info(f"Model saved to {path}")

    def save_losses(self, path):
        """
        Save the losses to a file.
        :param path: str, the file path to save the losses
        """
        print(self.losses)

        with open(path, 'wb') as f:
            pickle.dump(self.losses, f)

        # Save to txt file
        path = path.replace('.pkl', '.txt')
        path = path.replace('pkls', 'txts')
        with open(path, 'w') as f:
            for i, loss in enumerate(self.losses):
                f.write(f"Epoch {i + 1}: {loss}\n")
        self.logger.info(f"Losses saved to {path}")
    
    def load_model(self, path):
        """
        Load the trained model from a file.
        :param path: str, the file path to load the model
        """
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        if self.training_method == 'Basis' or self.training_method == 'SVD++' or self.training_method == 'AttributesSVD++':    
            self.num_factors = model_data['num_factors']
            self.learning_rate = model_data['learning_rate']
            self.reg_bias = model_data['reg_bias']
            self.epochs = model_data['epochs']
            self.μ = model_data['μ']
            self.b_x = model_data['b_x']
            self.b_i = model_data['b_i']
        if self.training_method == 'SVD++' or self.training_method == 'AttributesSVD++':
            self.reg_pq = model_data['reg_pq']
            self.P = model_data['P']
            self.Q = model_data['Q']
        if self.training_method == 'AttributesSVD++':
            self.reg_weight = model_data['reg_weight']
            self.attribute_weight = model_data['attribute_weight']
            self.item_attribute1, self.item_attribute2 = self.load_item_attributes(processed_attributes_path)

        self.logger.info(f"Model loaded from {path}")



if __name__ == "__main__":

    user_ratings = load_from_pickle('data/cache/pkls/user_ratings.pkl')
    # Initialize and train the SVD model
    num_factors = 100
    learning_rate = 0.005
    reg_bias = 0.02
    reg_pq = 0.02
    reg_weight = 0.02
    epochs = 1
    train_ratio = 0.8
    shuffle = True
    svd_method = "basis"
    training_method = "AttributesSVD++"




    svd = SVD(num_factors=num_factors, learning_rate=learning_rate, reg_bias=reg_bias, reg_pq=reg_pq, reg_weight=reg_weight, epochs=epochs, train_ratio=train_ratio, shuffle=shuffle, svd_method=svd_method, training_method=training_method)

    # Add the item attributes to the model
    svd.split_data(user_ratings)
    svd.fit(user_ratings)
    svd.evaluate()

    # Save the model on name includes all the hyperparameters
    model_name = f'{models_path}_{num_factors}_{learning_rate}_{reg_bias}_{reg_pq}_{reg_weight}_{epochs}_{train_ratio}_{shuffle}_{svd_method}_{training_method}.pkl'
    svd.save_model(model_name)
    losses_name = f'{models_path}_{num_factors}_{learning_rate}_{reg_bias}_{reg_pq}_{reg_weight}_{epochs}_{train_ratio}_{shuffle}_{svd_method}_{training_method}_losses.pkl'
    svd.save_losses(losses_name)



    # svd = SVD(training_method='AttributesSVD++')
    # svd.load_model('data/cache/pkls/model_20_0.005_0.02_0.02_0.02_30_basis_AttributesSVD++.pkl')
    # # print所有参数
    # print(svd.num_factors)
    # print(svd.learning_rate)
    # print(svd.reg_bias)
    # print(svd.reg_pq)
    # print(svd.reg_weight)
    # print(svd.epochs)
    # print(svd.μ)
    # print(svd.b_x)
    # print(svd.b_i)
    # print(svd.P)
    # print(svd.Q)
    # print(svd.attribute_weight)
    # model_data = load_from_pickle('data/cache/pkls/model_20_0.005_0.02_0.02_0.02_30_basis_AttributesSVD++.pkl')
    # num_factors = model_data['num_factors']
    # learning_rate = model_data['learning_rate']
    # reg_bias = model_data['reg_bias']
    # reg_pq = model_data['reg_pq']
    # epochs = model_data['epochs']
    # svd_method = "basis"
    # training_method = "AttributesSVD++"

    # print(model_data['num_factors'])
    # print(model_data['learning_rate'])
    # print(model_data['reg_bias'])
    # print(model_data['epochs'])
    # print(model_data['μ'])
    # print(model_data['b_x'])
    # print(model_data['b_i'])
    # print(model_data['P'])
    # print(model_data['Q'])
    # svd = SVD(num_factors=model_data['num_factors'], 
    #           learning_rate=model_data['learning_rate'], 
    #           reg_bias=model_data['reg_bias'], 
    #           reg_pq=model_data['reg_pq'], 
    #           reg_weight=0.02, epochs=1, 
    #           svd_method='basis', 
    #           training_method='AttributesSVD++')
    # svd.μ = model_data['μ']
    # svd.b_x = model_data['b_x']
    # svd.b_i = model_data['b_i']
    # svd.P = model_data['P']
    # svd.Q = model_data['Q']

    # svd.fit(user_ratings)

    # # Save the model on name includes all the hyperparameters
    # model_name = f'{models_path}_{num_factors}_{learning_rate}_{reg_bias}_{reg_pq}_{0.02}_{epochs}_{svd_method}_{training_method}.pkl'
    # svd.save_model(model_name)
    # losses_name = f'{models_path}_{num_factors}_{learning_rate}_{reg_bias}_{reg_pq}_{0.02}_{epochs}_{svd_method}_{training_method}_losses.pkl'
    # svd.save_losses(losses_name)

    # # Example prediction
    # user_id = 0
    # item_id = 378216
    # rating = svd.predict(user_id, item_id)
    # print(f'Predicted rating for user {user_id} and item {item_id}: {rating}')




    svd = SVD(training_method='AttributesSVD++')
    svd.split_data(user_ratings)
    svd.load_model('data\cache\pkls\model_100_0.005_0.02_0.02_0.02_1_0.8_True_basis_AttributesSVD++.pkl')

    # Evaluate the model
    rmse = svd.evaluate()
    print(f'RMSE on evaluation set: {rmse}')