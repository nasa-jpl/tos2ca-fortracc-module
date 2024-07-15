import warnings

# Create a deprecation decorator for python 3.8
def deprecated(message):
    def _deprecated(obj):
        warnings.simplefilter(
            'always',
            DeprecationWarning
        )
        warnings.warn(
            message=message,
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter(
            'default',
            DeprecationWarning
        )
        return obj
    return _deprecated
