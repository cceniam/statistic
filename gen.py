import numpy as np

def generate_data(size, mean1, mean2, std1, std2):
    """
    Generate data for two datasets
    :param size:  Number of samples
    :param mean1:  Mean of the first dataset
    :param mean2:  Mean of the second dataset
    :param std1:  Standard deviation of the first dataset
    :param std2:  Standard deviation of the second dataset
    :return:  Two datasets
    """
    data1 = np.random.normal(mean1, std1, size).round().astype(int)
    data2 = np.random.normal(mean2, std2, size).round().astype(int)
    return data1, data2