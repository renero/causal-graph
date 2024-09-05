"""
This Python function maintains the same structure and functionality as the original 
R function:

1. The function is named `train_gp` and takes three parameters: `X`, `y`, and `pars` 
(with a default empty dictionary).
2. It raises a `NotImplementedError` with the message "GP regression not implemented."
3. It returns `None` (which is equivalent to R's `NULL`).

Note that in Python, we use `raise` instead of `stop()` to throw exceptions, and 
we use `NotImplementedError` as it's the most appropriate built-in exception for 
this case. Also, the default value for `pars` is set to `None` and then initialized 
as an empty dictionary inside the function, which is a common Python idiom to 
avoid mutable default arguments.
"""


def train_gp(X, y, pars=None):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        pars (_type_, optional): _description_. Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if pars is None:
        pars = {}

    raise NotImplementedError('GP regression not implemented.')
