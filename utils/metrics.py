import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Add list of metrix to compute

# TODO: 
# 1. ADD metric docs.
# 2. Add mask filling, like nan replacement, or add nan support for metrics
# 3. Obj identification and detection. (Cold pools, img processing,
# bounding regions, what are the properties inside outisde prob detection,
# detection of features, quatification of features)
# 4. Symmetric KL divergence and Jensen-Shannon divergence

class MatrixComparer:
    metrics = {}

    @classmethod
    def register_metric(cls, name, func):
        cls.metrics[name] = func

    @classmethod
    def compute_metrics(cls, mat1, mat2):
        results = {}
        for name, func in cls.metrics.items():
            results[name] = func(mat1, mat2)
        return results

# Mean of the squared differences between two matrices.
def mean_squared_error(mat1, mat2):
    return np.mean((mat1 - mat2) ** 2)

# Kullback-Leibler divergence between two matrices. Assumes the matrices are probability distributions.
# TODO: Add normalization of the matrices to sum to 1.
def kl_divergence(mat1, mat2):
    mat1_prob = mat1 / np.sum(mat1)
    mat2_prob = mat2 / np.sum(mat2)
    return entropy(mat1_prob.flatten(), mat2_prob.flatten())

# Root of the mean of the squared differences between two matrices.
def root_mean_squared_error(mat1, mat2):
    return np.sqrt(np.mean((mat1 - mat2) ** 2))

# Mean of the absolute differences between two matrices, also known as L1 loss.
def mean_absolute_error(mat1, mat2):
    return np.mean(np.abs(mat1 - mat2))

# Cosine similarity between two matrices, computed as 1 - cosine distance.
def cosine_similarity(mat1, mat2):
    return 1 - cosine(mat1.flatten(), mat2.flatten())

def normalized_root_mean_squared_error(mat1, mat2):
    range_of_values = np.max(mat1) - np.min(mat1)
    return root_mean_squared_error(mat1, mat2) / range_of_values if range_of_values != 0 else np.inf

def correlation_coefficient(mat1, mat2):
    corr, _ = pearsonr(mat1.flatten(), mat2.flatten())
    return corr

def peak_signal_to_noise_ratio(mat1, mat2):
    mse = np.mean((mat1 - mat2) ** 2)
    if mse == 0:
        return float('inf')  # Avoid division by zero if there is no noise
    max_pixel = 1.0  # Assuming the image pixel values are in the range [0,1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Sorensen-Dice coefficient. Measures the overlap between two binary matrices.
def dice_coefficient(mat1, mat2):
    intersection = np.sum(mat1 * mat2)
    total = np.sum(mat1) + np.sum(mat2)
    dice = 2 * intersection / total if total != 0 else 1
    return dice

# Intersection over union. Measures the overlap between two binary matrices.
def jaccard_index(mat1, mat2):
    intersection = np.sum(mat1 * mat2)
    union = np.sum(mat1 + mat2 > 0)
    jaccard = intersection / union if union != 0 else 1
    return jaccard

# Hamming distance. Measures the proportion of bits that differ between two binary matrices.
def hamming_distance(mat1, mat2):
    return np.sum(mat1 != mat2) / mat1.size

# F1 score. Measures the balance between precision and recall.
def f1_score(mat1, mat2):
    precision = np.sum(mat1 * mat2) / np.sum(mat2) if np.sum(mat2) != 0 else 0
    recall = np.sum(mat1 * mat2) / np.sum(mat1) if np.sum(mat1) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1

# Register metrics
MatrixComparer.register_metric('mean_squared_error', mean_squared_error)
MatrixComparer.register_metric('kl_divergence', kl_divergence)
MatrixComparer.register_metric('root_mean_squared_error', root_mean_squared_error)
MatrixComparer.register_metric('mean_absolute_error', mean_absolute_error)
MatrixComparer.register_metric('cosine_similarity', cosine_similarity)
MatrixComparer.register_metric('normalized_root_mean_squared_error', normalized_root_mean_squared_error)
MatrixComparer.register_metric('correlation_coefficient', correlation_coefficient)
MatrixComparer.register_metric('peak_signal_to_noise_ratio', peak_signal_to_noise_ratio)
MatrixComparer.register_metric('dice_coefficient', dice_coefficient)
MatrixComparer.register_metric('jaccard_index', jaccard_index)
MatrixComparer.register_metric('hamming_distance', hamming_distance)
MatrixComparer.register_metric('f1_score', f1_score)


# Usage example
# mat1 = np.random.rand(10, 10)
# mat2 = np.random.rand(10, 10)

# # Compute metrics
# results = MatrixComparer.compute_metrics(mat1, mat2)
# print(results)
