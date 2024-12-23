import numpy as np
from scipy.stats import ttest_ind

import gen

if __name__ == '__main__':
    # Parameters
    size = 109
    mean1, mean2 = 21.43, 16.43
    std1, std2 = 3.15, 2.91
    t = 7.928
    # Generate data
    data1, data2 = gen.generate_data(size, mean1, mean2, std1, std2, t)
    # Output results
    # Calculating the mean and standard deviation for both datasets
    mean1_actual = np.mean(data1)
    std1_actual = np.std(data1)
    mean2_actual = np.mean(data2)
    std2_actual = np.std(data2)

    # Performing the t-test
    t_statistic, p_value = ttest_ind(data1, data2)
    print("First Dataset: Mean =", mean1_actual, ", Standard Deviation =", std1_actual)
    print("Second Dataset: Mean =", mean2_actual, ", Standard Deviation =", std2_actual)
    print("T-statistic =", t_statistic, ", P-value =", p_value)
    print(data1)
    print(data2)
