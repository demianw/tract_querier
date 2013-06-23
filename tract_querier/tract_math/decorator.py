__all__ = ['tract_math_operation']


def tract_math_operation(help_text, needs_one_tract=True):
    '''
    Decorator to identify tract_math functionalities the name of the
    function will be automatically incorporated to the tract_math options

    Parameters
    ----------
    help_text: help for the operation
    needs_one_tract: tells the script if all the input tractographies should
                      be unified as one or left as a tractography list
    '''

    def internal_decorator(func):
        func.help_text = help_text
        func.needs_one_tract = needs_one_tract
        return func

    return internal_decorator
