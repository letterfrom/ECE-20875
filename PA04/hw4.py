###################################
# Problem 1: Higher-order functions
###################################

def conditional_map(predicate, if_func, else_func, L):
    '''
    Returns a new list R where each element in R is if_func(element) if predicate(element) 
    returns True, and else_func(element) otherwise.

    :param predicate: function that takes an element and returns True/False
    :param if_func: function to apply to elements where predicate returns True
    :param else_func: function to apply to elements where predicate returns False
    :param L: list of elements
    :return R: list of results
    '''
    # Fill in
    pass

def compose_map(func_1, func_2, L):
    """
    Returns a new list R where each element in R is fun2(fun1(i)) for the
    corresponding element i in L
    :param fun1: function
    :param fun2: function
    :param L: list
    :return R: list
    """
    # Fill in
    pass

def compose(func_1, func_2):
    # Fill in
    """
    Returns a new function ret_fun. ret_fun should take a single input i, and return
    fun1(fun2(i))
    :param fun1: function
    :param fun2: function
    :return ret_fun: function
    """
    def ret_fun(i):
        # Fill in
        pass

    return ret_fun

def repeater(fun, num_repeats):
    """
    Returns a new function ret_fun. This takes in a list of functions `funlist` and a list of integers `num_repeats`, 
    and returns a new function, `ret_fun`. The new function takes an input `x` and calls the 
    first function in `funlist` repeated a number of times equal to the first number 
    in the list `num_repeats`, and then calls the second function in `funlist` repeated 
    a number of times equal to the second number in the list `num_repeats`, continuing this
    pattern until the end of `funlist` is reached.
    :param fun: list of functions
    :param num_repeats: list of int
    :return ret_fun: function
    """
    def ret_fun(x):
        # Fill in
        pass

    return ret_fun

#############################################
# Problem 2: Stencil and Box filter functions
#############################################

def stencil(data, f, width):
    """
    1) perform a stencil using the filter function f with 'width', on list data.
    2) return the resulting list output.
    3) note that if len(data) is k, len(output) would be k - width + 1.
    4) f will accept input a list of size 'width' and return a single number.

    :param data: list
    :param f: function
    :param width: int
    :return output: list
    """
    # Fill in
    pass


def create_box(box):
    """
    1) This function takes in a list, box.
    The box_filter function defined below accepts a list L of length len(box) and returns a simple
    convolution of it with the list, box.

    2) The meaning of this box filter is as follows:
    for each element of input list L, multiply L[i] by box[len(box) - 1  - i],
    sum the results of all of these multiplications and return the sum.

    3) For a box of length 3, box_filter(L) should return:
      (box[2] * L[0] + box[1] * L[1] + box[0] * L[2]),
      similarly, for a box of length 4, box_filter should return:
      (box[3] * L[0] + box[2] * L[1] + box[1] * L[2] + box[0] * L[3])

    The function create_box returns the box_filter function, as well as the length
    of the input list box

    :param box: list
    :return box_filter: function, len(box): int
    """
    # Fill in
    def box_filter(L):
        # Fill in
        pass

    return box_filter, len(box)

if __name__ == '__main__':
    # To test your functions, run: python given_tests.py
    # This will import your hw4.py and test all functions
    print("To test your homework functions, please run: python given_tests.py")
