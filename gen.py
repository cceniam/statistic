import numpy as np
from scipy.stats import ttest_ind
import concurrent.futures
import multiprocessing
import time
import os

# Set a fixed random seed for reproducibility
np.random.seed(42)


def generate_data_worker(data1, data2, t_value, max_iterations):
    """
    Worker function to iteratively adjust data2 to match the target t-value
    :param data1: First dataset
    :param data2: Second dataset to be adjusted
    :param t_value: Target t-value
    :param max_iterations: Maximum number of iterations to adjust data to match t-value
    :return: Adjusted data2
    """
    learning_rate = 0.001  # Further reduced initial learning rate
    min_learning_rate = 0.0001  # Minimum learning rate to avoid too small steps
    max_learning_rate = 0.1  # Increased maximum learning rate for faster convergence
    noise_scale = 0.05  # Reduced initial noise scale
    for iteration in range(max_iterations):
        current_t, _ = ttest_ind(data1, data2)

        # If the current t-value is close enough to the target t-value, stop
        if np.isclose(current_t, t_value, atol=0.01):
            print(f"Early stopping at iteration {iteration} with t-value: {current_t}")
            return data2

        # Adjust the second dataset to move towards the target t-value using an adaptive learning rate
        adjustment = (t_value - current_t) * learning_rate
        adjustment = np.clip(adjustment, -0.2, 0.2)  # Clip adjustments to avoid large jumps
        random_noise = np.random.normal(0, noise_scale * np.abs(t_value - current_t) / (iteration + 1),
                                        size=len(data2))  # Gradually reduce noise
        data2 = data2 + adjustment + random_noise

        # Adapt the learning rate to improve convergence
        if np.abs(current_t - t_value) > 1.0:
            learning_rate = max(min_learning_rate, learning_rate * 0.9)
        else:
            learning_rate = min(max_learning_rate, learning_rate * 1.2)

        # Gradually reduce noise scale to make adjustments more precise over time
        noise_scale = max(0.005, noise_scale * 0.95)

    return data2


def generate_data(size, mean1, mean2, std1, std2, t_value, max_iterations=5000, num_threads=None):
    """
    Generate data for two datasets with a target t-value using multiprocessing
    :param size: Number of samples
    :param mean1: Mean of the first dataset
    :param mean2: Mean of the second dataset
    :param std1: Standard deviation of the first dataset
    :param std2: Standard deviation of the second dataset
    :param t_value: Target t-value
    :param max_iterations: Maximum number of iterations to adjust data to match t-value
    :param num_threads: Number of threads to use for finding the solution
    :return: Two datasets
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    # Initial random data generation
    data1 = np.random.normal(mean1, std1, size)

    adjusted_data2 = None
    if __name__ == "__main__":
        for _ in range(10):  # Add a loop to attempt multiple times until a good result is found
            # Use multiprocessing to adjust data2 to match the target t-value
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(generate_data_worker, data1,
                                    np.random.RandomState(np.random.randint(0, 1e6)).normal(mean2, std2, size), t_value,
                                    max_iterations)
                    for i in range(num_threads)
                ]

                for future in concurrent.futures.as_completed(futures):
                    adjusted_data2 = future.result()
                    current_t, _ = ttest_ind(data1, adjusted_data2)

                    # If the current t-value is close enough to the target t-value, stop and return
                    if np.isclose(current_t, t_value, atol=0.01):
                        print(f"Successful adjustment with t-value: {current_t}")
                        return data1, adjusted_data2

    # If none of the threads succeeded, return the last adjusted dataset
    print("Failed to reach the target t-value within tolerance after multiple attempts.")
    return data1, adjusted_data2


# Example usage
if __name__ == "__main__":
    data1, data2 = generate_data(size=109, mean1=21.43, mean2=16.43, std1=3.15, std2=2.91, t_value=7.928)
    print("Dataset 1:", data1)
    print("Dataset 2:", data2)

    # Verify the t-value
    t_stat, p_value = ttest_ind(data1, data2)
    print("Generated t-value:", t_stat)
#     size = 109
#     mean1, mean2 = 21.43, 16.43
#     std1, std2 = 3.15, 2.91
#     t = 7.928