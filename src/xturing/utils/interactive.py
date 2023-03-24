def is_interactive_execution():
    """Checks if the current code is executed in a notebook or using the Python interpreter"""

    is_interactive = None
    try:
        get_ipython().__class__.__name__
        is_interactive = True
    except:
        is_interactive = False

    return is_interactive
