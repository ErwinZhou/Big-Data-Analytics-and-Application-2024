from utils.config import pickle

def save_to_pickle(obj, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        print(f"An error occurred while trying to save data to {path}: {e}")
        return False

def load_from_pickle(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"An error occurred while trying to load data from {path}: {e}")
        return False