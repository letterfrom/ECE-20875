# given_tests.py
# Test functions provided to students for testing their implementations

from hw4 import *

def round_list(numbers, decimals=4):
    """Helper function to round a list of numbers to specified decimal places"""
    return [round(num, decimals) for num in numbers]

def run_problem1_tests(data):
    # Problem 1 test functions. You can modify the data and functions to test other conditions.
    def test1(x):
        return x * 3

    def test2(x):
        return x - 1

    def is_positive(x):
        return x > 0

    # Testing the conditional_map function
    print(conditional_map(is_positive, test1, test2, data)) # Apply test1 to positive numbers, test2 to non-positive numbers

    # Testing the compose_map function
    print(compose_map(test1, test2, data)) # Calling the compose_map function with functions test1, test2 and the list data as the argument

    print(compose_map(test2, test1, data)) # Calling the compose_map function with functions test2, test1 and the list data as the argument (order of function input changed)

    # Using the compose function with functions test1 and test2 as arguments and returning its value to f1
    f1 = compose(test1, test2)

    # Running f1 with i=4
    print(f1(4))

    # Applying f1 on to each value in the list data
    print(list(map(f1, data)))

    # Using the compose function with functions test2 and test1 as arguments and returning its value to f2
    f2 = compose(test2, test1)

    # Running f2 with i=4
    print(f2(4))

    # Applying f2 on to each value in the list data
    print(list(map(f2, data)))

    # Testing the repeater function that with the function test1 with different num_repeats argument.
    z = repeater([test1, test2], [0 ,0])
    once = repeater([test1, test2], [1, 1])
    twice = repeater([test1, test2], [2, 2])
    thrice = repeater([test1, test2], [3, 3])

    print("repeat 0 times: {}".format(z(5)))
    print("repeat 1 time: {}".format(once(5)))
    print("repeat 2 times: {}".format(twice(5)))
    print("repeat 3 times: {}".format(thrice(5)))


def run_problem2_tests(data):
    # Problem 2 test functions. You can modify the data and functions to test other conditions.
    def mov_avg(L):
        # The `mov_avg` function takes in a list `L` and returns the moving average of its elements.
        return float(sum(L)) / 3
    
    # Define a function `sum_sq` that takes in a list `L` and returns the sum of the squares of the elements in `L`.
    def sum_sq(L):
        return sum([i ** 2 for i in L])
    
    print(round_list(stencil(data, mov_avg, 2)))
    print(stencil(data, sum_sq, 5))
    
    # note that this creates a moving average!
    box_f1, width1 = create_box([1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0/4])
    print(round_list(stencil(data, box_f1, width1)))
    
    box_f2, width2 = create_box([-0.2, -0.3, 0.3, 0.2])
    print(round_list(stencil(data, box_f2, width2)))


if __name__ == '__main__':
    # Test your homework implementation
    data = [2,4,6,8,10,10,-3,-6,-7]
    
    print("=== Problem 1 Tests ===")
    run_problem1_tests(data)
    
    print("\n=== Problem 2 Tests ===")
    run_problem2_tests(data)