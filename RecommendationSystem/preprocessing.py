"""
Data Preprocessing
"""

from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import numpy as np
from config import *


def save_sparse_matrix(file_path: str, matrix: csr_matrix):
    """
    Save the sparse matrix to the file.
    :param file_path: the path of the file
    :param matrix: the sparse matrix
    """
    save_npz(file_path, matrix)


def load_sparse_matrix(file_path: str) -> csr_matrix:
    """
    Load the sparse matrix from the file.
    :param file_path: the path of the file
    :return: the sparse matrix
    """
    return load_npz(file_path)


def read_data(file_path: str) -> dict:
    """
    Read the data from the file.
    file format:
    <user id>|<numbers of rating items>
    <item id>   <score>

    :param file_path: the path of the file
    :return: {item_id: [(user_id, score), ...], ...}
    """
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            user_id, num_items = line.strip().split('|')
            user_id, num_items = int(user_id), int(num_items)
            for _ in range(num_items):
                item_id, score = file.readline().strip().split()
                item_id, score = int(item_id), int(score)
                data.setdefault(item_id, []).append((user_id, score))
    return data


def create_item_index_mapping(data: dict) -> (dict, dict):
    """
    Create the mapping from the original discontinuous item ids to the continuous sequence.

    NOTE:
        - Since some items may not have any ratings (New Items), their item ids should be neglected for efficiency,
    thus making the item id sequence discontinuous.
        - To handle this, this function creates a mapping that maps the item ids to the continuous sequence. The
    mapping is stored in the dictionary `item_mapping` and its reverse mapping is stored in the dictionary
    `reverse_mapping`.
        - This mapping is order-preserving.

    :param data: data in the format of {item_id: [(user_id, score), ...], ...}
    :return: item_mapping(item_id: index), reverse_mapping(index: item_id)
    """
    item_mapping = {}
    reverse_mapping = {}
    sorted_keys = sorted(data.keys())  # Sort the item ids
    for index, item_id in enumerate(sorted_keys):
        item_mapping[item_id] = index
        reverse_mapping[index] = item_id
    return item_mapping, reverse_mapping


def item_index_mapping(data: dict, item_mapping: dict, allow_expansion: bool = True) -> (dict, dict):
    """
    Map item indexes using the given `item_mapping`.
    NOTE:
        - This function can also be used to restore item indexes to the original item ids using the `reverse_mapping`.
        - Actually, the original `item_mapping` will also be modified if the `allow_expansion` is set to True. But using
    the returned value is recommended and can prominently alert the user.

    :param data: data in the format of {item_id: [(user_id, score), ...], ...}
    :param item_mapping: the mapping used to convert indexes
    :param allow_expansion: whether to allow the auto expansion of the mapping
    :return: the mapped data in the format of {index: [(user_id, score), ...], ...}, the auto-expanded mapping
    """
    mapped_data = {}
    expand_index = len(item_mapping)
    for item_id, ratings in data.items():
        index = item_mapping.get(item_id, None)
        if index is not None:
            mapped_data[index] = ratings
        elif allow_expansion:
            item_mapping[item_id] = expand_index
            expand_index += 1
        else:
            print('Warning: Item {} is not in the mapping! Ignored and proceeding.'.format(item_id))
    return mapped_data, item_mapping


def get_reverse_mapping(mapping: dict) -> dict:
    """
    Get the reverse mapping of the given mapping.
    :param mapping: the mapping
    :return: the reverse mapping
    """
    reverse_mapping = {}
    for key, value in mapping.items():
        reverse_mapping[value] = key
    return reverse_mapping


def data_normalization(data: dict, scale: float) -> dict:
    """
    Normalize the data.
    First, transform the scores using $ ln(1+x) $ to handle the long-tail distribution.
    Then, normalize the scores to [0, scale].
    :param data: data in the format of {item_id: [(user_id, score), ...], ...}
    :param scale: the scale factor
    :return: the normalized data in the format of {item_id: [(user_id, score), ...], ...}
    """
    normalized_data = {}
    for item_id, ratings in data.items():
        scores = np.log1p([score for _, score in ratings])  # ln(1+x)
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:  # All the scores are the same
            normalized_data[item_id] = [(user_id, scale / 2) for user_id, _ in ratings]
        else:
            normalized_data[item_id] = [
                (user_id, scale * (score - min_score) / (max_score - min_score)) for user_id, score in ratings
            ]
    return normalized_data


def data_centering(data: dict) -> dict:
    """
    Center the data: every row (item) subtracting the mean rating of each item.
    :param data: data in the format of {item_id: [(user_id, score), ...], ...}
    :return: the centered data in the format of {item_id: [(user_id, score), ...], ...}
    """
    centered_data = {}
    for item_id, ratings in data.items():
        scores = [score for _, score in ratings]
        mean_score = sum(scores) / len(scores)
        centered_data[item_id] = [(user_id, score - mean_score) for user_id, score in ratings]
    return centered_data


def data2matrix(data: dict) -> csr_matrix:
    """
    Convert the data to the item-user matrix.
    :param data: data in the format of {item_id: [(user_id, score), ...], ...}
    :return: the item-user matrix of shape (# of item_ids, MAX_USER_NUM)
    """
    item_matrix = csr_matrix((len(data), MAX_USER_NUM), dtype=float)
    item_matrix = lil_matrix(item_matrix)  # faster for row-wise operations
    sorted_keys = sorted(data.keys())  # Sort the item ids
    for item_id in sorted_keys:
        for user_id, score in data[item_id]:
            item_matrix[item_id, user_id] = score
    item_matrix = item_matrix.tocsr()  # Convert to CSR format for fast matrix operations
    return item_matrix


def read_item_attr(file_path: str) -> dict:
    """
    Read the item attributes from the file.

    file format:
    <item id>|<attribute_1>|<attribute_2>

    NOTE:
        'None' means this item does not belong to any of attribute_1 or attribute_2.

    :param file_path: the path of the file
    :return: the item attributes in the format of {item_id: (attribute_1, attribute_2), ...}
    """
    item_attr = {}
    with open(file_path, 'r') as file:
        for line in file:
            item_id, attr_1, attr_2 = line.strip().split('|')
            item_id = int(item_id)
            if attr_1 == 'None':
                attr_1 = None
            else:
                attr_1 = int(attr_1)
            if attr_2 == 'None':
                attr_2 = None
            else:
                attr_2 = int(attr_2)
            item_attr[item_id] = (attr_1, attr_2)
    return item_attr


if __name__ == '__main__':
    import os
    # Read the data
    train_data = read_data(str(os.path.join(DATA_DIR, TRAIN_FILE)))
    item_mapping, reverse_mapping = create_item_index_mapping(train_data)
    train_data, item_mapping = item_index_mapping(train_data, item_mapping, allow_expansion=False)
    # Normalize and center the data
    train_data = data_normalization(train_data, scale=DATA_NORM_SCALE_FACTOR)
    train_data = data_centering(train_data)
    # Convert the data to the item-user matrix
    item_matrix = data2matrix(train_data)
    # Save the item-user matrix
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    save_sparse_matrix(str(os.path.join(CACHE_DIR, ITEM_MATRIX_FILE)), item_matrix)
    print('The item-user matrix has been saved to {}.'.format(str(os.path.join(CACHE_DIR, ITEM_MATRIX_FILE))))
    print('Data Preprocessing Finished!')

    # Read the item attributes
    # item_attr = read_item_attr(str(os.path.join(DATA_DIR, ITEM_ATTR_FILE)))

    pass
