from utils.config import np, pd


def SSE(y_true, y_pred):
    """
    The function to calculate the Sum of Squared Errors
    Args:
        y_true: The true ratings
        y_pred: The predicted ratings
    Returns:
        The Sum of Squared Errors
    """
    return np.sum((y_true - y_pred) ** 2)

def RMSE(y_true, y_pred):
    """
    The function to calculate the Root Mean Squared Error
    Args:
        y_true: The true ratings
        y_pred: The predicted ratings
    Returns:
        The Root Mean Squared Error
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))