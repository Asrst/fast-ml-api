import random
from typing import Any, Dict, List, Tuple, NoReturn, Sequence


def calc_freq(input_list:List[int]) -> Dict[int, int]:

    """
    Return Dict with Count of each number present
    input List

    Parameters
    ----------
    input_list : List[int]
        List of integers

    Returns
    -------
    Dict[int, int]

    Raises
    ------
    TypeError
        when input is not List.
    """

    if not isinstance(input_list, list):
        error_msg = f"input_list must be List of Integers (type: List[int])."
        raise TypeError(error_msg)

    result:Dict[int, int] = {}

    for integer in input_list:
        result[integer] = result.get(integer, 0) + 1
    
    return result


if __name__ == "__main__":
    # Below example will give the frequency counts of all elements.
    integers_list = [random.randint(0, 5) for i in range(0, 10)]
    print('Test Case 1:')
    r = calc_freq(integers_list)
    print(f"Input :{integers_list}.")
    print(f"Output :{r}.")

    # Below example will raise an exception due to wrong input type.
    print('\nTest Case 2:')
    integers_tuple = tuple(integers_list)
    r = calc_freq(integers_tuple)