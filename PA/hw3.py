import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(histogram):
    """
    takes a list of counts and converts to a list of probabilities, outputs the probability list.
    :param histogram: a numpy ndarray object
    :return: list
    """
    counts = np.asarray(histogram, dtype=float)
    total = counts.sum()
    if total == 0:
        # No samples: return zeros of the same length
        return [0.0] * len(counts)
    probs = counts / total
    return probs.tolist()


def compute_j(histogram, bin_width, num_samples):
    """
    takes list of counts, uses norm_histogram function to output the histogram of probabilities,
    then calculates compute_j for one specific bin width (reference: histogram.pdf page19)
    :param histogram: list
    :param bin_width: float
    :param num_samples: int
    :return: float
    """
    # Normalize to probabilities
    p = np.asarray(norm_histogram(histogram), dtype=float)
    n = int(num_samples)
    h = float(bin_width)

    # If degenerate inputs, return +inf to avoid selecting this bin width
    if h <= 0 or n <= 1:
        return float("inf")

    # LSCV-like cost for histogram densities
    # J(h) = 2/((n-1)h) - ((n+1)/((n-1)h)) * sum p_i^2
    sum_p2 = float(np.sum(p * p))
    j = (2.0 / ((n - 1) * h)) - ((n + 1.0) / ((n - 1) * h)) * sum_p2
    return j


def sweep_n(data, min_val, max_val, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    The variable "data" is the raw data that still needs to be "processed"
    with matplotlib.pyplot.hist to output the histogram

    You must utilize the variables (data, min_val, max_val, min_bins, max_bins)
    in your code for 'sweep_n' to determine the correct input to the function 'matplotlib.pyplot.hist',
    specifically the values to (x, bins, range).
    Other input variables of 'matplotlib.pyplot.hist' can be set as default value.

    :param data: list
    :param min_val: int
    :param max_val: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    x = np.asarray(data)
    n = len(x)
    lo = float(min_val)
    hi = float(max_val)

    # Ensure bounds are valid; if not, return an array of infs
    if hi <= lo:
        return [float("inf")] * max(0, (max_bins - min_bins + 1))

    js = []
    # We'll suppress plotting by using plt.hist and immediately clearing artists
    for bins in range(int(min_bins), int(max_bins) + 1):
        # bin width from range and bin count
        h = (hi - lo) / float(bins)
        # Use the required API: matplotlib.pyplot.hist with (x, bins, range)
        counts, edges, patches = plt.hist(x, bins=bins, range=(lo, hi))
        # Compute J for this binning
        j = compute_j(counts, h, n)
        js.append(j)
        # Clean up the created artists to avoid accumulating figures
        for p in patches:
            p.remove()
    return js


def find_min(l):
    """
    takes a list of numbers and returns the three smallest number in that list and their index.
    return a dict i.e.
    {index_of_the_smallest_value: the_smallest_value, index_of_the_second_smallest_value: the_second_smallest_value, ...}

    For example:
        A list(l) is [14,27,15,49,23,41,147]
        Then you should return {0: 14, 2: 15, 4: 23}

    :param l: list
    :return: dict: {int: float}
    """
    arr = list(l)
    # Pair (value, index), sort by value then index for determinism, take up to 3
    triples = sorted(((v, i) for i, v in enumerate(arr)), key=lambda t: (t[0], t[1]))[:3]
    return {i: v for (v, i) in triples}


# ============================== P2 ==================================

import scipy.stats as stats
import numpy as np


def get_data(filename):
    return np.loadtxt(filename)


def get_coordinates(data, each_dist):
    # Part B
    """
    calculates the QQ plot given an array of data and a name of a distribution
    outputs a tuple of 2 numpy arrays from the output of the QQ plot
    :param data: np.ndarray
    :param each_dist: str
    :return: (np.ndarray, np.ndarray)
    """
    # Your code starts here...
    
    QQ = stats.probplot(data, dist=each_dist, plot=None)
    
    return tuple(QQ[0])


def calculate_distance(x, y):
    # Part B
    """
    calculates the projected distance between x and y
    returns the distance as a float
    :param x: float
    :param y: float
    :return: float
    """
    # Your code starts here...

    distance = np.sqrt((x - ((x + y) / 2))**2 + (y - ((x + y) / 2))**2)

    return distance


def find_dist(data):
    # Part B
    """
    from a dictionary of distribution names and their respective errors, finds the distribution having the minimum value
    outputs the minimum value and the name of the distribution
    (NOTE the output is NOT structured as a tuple)
    :param data: dict: {str: float}
    :return: str, float
    """
    # Your code starts here...
    
    min_dist = min(data, key=data.get)
    min_err = data[min_dist]

    return str(min_dist), float(min_err) #forcing the output to be a string and float, respectively


def main(data_file):
    """
    Input a csv file and return distribution type, the error corresponding to the distribution type (e.g. return ('norm', 0.32))
    :param: *.csv file name (str)
    :return: (str, float)
    """
    data = get_data(data_file)
    dists = ("norm", "expon", "uniform", "wald")
    sum_err = [0] * 4
    for ind, each_dist in enumerate(dists):
        X, Y = get_coordinates(data, each_dist)
        for x, y in zip(X, Y):
            sum_err[ind] += calculate_distance(x, y)
    return find_dist(dict(zip(dists, sum_err)))


if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
  ############### Uncomment for P2 #################

    # for each_dataset in [
    #     "sample_norm.csv",
    #     "sample_expon.csv",
    #     "sample_uniform.csv",
    #     "sample_wald.csv",
    #     "distA.csv",
    #     "distB.csv",
    #     "distC.csv",
    # ]:
    #     print(main(each_dataset))
