from preprocessing.cache import save_to_pickle, load_from_pickle


def generate_index_map(train_data_path):
    """
    Because the sequence of the items is not continuous
    A index map is needed to map the item id to a continuous index
    Finally, the index map will be dumped to item_idx.pkl
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
    # Dump the index map to index pickle file
    if save_to_pickle(index_map, 'data/cache/item_idx.pkl'):
        return True
    return False
