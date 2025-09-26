###################################
# Problem 1: Higher-order functions
###################################

def conditional_map(predicate, if_func, else_func, L):
    '''
    Returns a new list R where each element in R is if_func(element) if predicate(element) 
    returns True, and else_func(element) otherwise.
    '''
    R = []
    for x in L:
        if predicate(x):
            result = if_func(x)
        else:
            result = else_func(x)
        R.append(result)
    return R


def compose_map(func_1, func_2, L):
    """
    Returns a new list R where each element in R is func_2(func_1(i)) for the
    corresponding element i in L
    """
    R = []
    for i in L:
        temp = func_1(i)
        result = func_2(temp)
        R.append(result)
    return R


def compose(func_1, func_2):
    """
    Returns a new function ret_fun. ret_fun should take a single input i, and return
    func_1(func_2(i))
    """
    def ret_fun(i):
        temp = func_2(i)
        result = func_1(temp)
        return result

    return ret_fun


def repeater(fun, num_repeats):
    """
    Returns a new function ret_fun. This takes in a list of functions `funlist` and a list of integers `num_repeats`, 
    and returns a new function, `ret_fun`. The new function takes an input `x` and calls the 
    first function in `funlist` repeated a number of times equal to the first number 
    in the list `num_repeats`, and then calls the second function in `funlist` repeated 
    a number of times equal to the second number in the list `num_repeats`, continuing this
    pattern until the end of `funlist` is reached.
    """
    def ret_fun(x):
        for f, n in zip(fun, num_repeats):
            for _ in range(n):
                x = f(x)
        return x

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
    """
    # Fill in
    output = []
    k = len(data)
    limit = k - width + 1
    for start in range(limit):
        window = []
        for j in range(width):
            window.append(data[start + j])
        value = f(window)
        output.append(value)
    return output


def create_box(box):
    """
    Returns a stencil function (box_filter) and the box length.
    The box_filter performs a 1-D convolution with the *flipped* box.
    """
    # Fill in
    n = len(box)

    def box_filter(L):
        # Fill in
        if len(L) != n:
            print(f"Calling box filter with the wrong length list. Expected length of list should be {n}.")
            return 0
        total = 0
        for i in range(n):
            total += box[n - 1 - i] * L[i]
        return total

    return box_filter, len(box)

if __name__ == '__main__':
    # To test your functions, run: python given_tests.py
    # This will import your hw4.py and test all functions
    print("To test your homework functions, please run: python given_tests.py")
