"""General utilities."""

import functools
import operator
import warnings
from typing import Type, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Num
from numpy.typing import ArrayLike

# jax.typing.ArrayLike and jaxtyping.ArrayLike don't include eg tuples,lists,iterables
# like np.ArrayLike. This combines all the usual array types
Arrayish = Union[Array, ArrayLike]


def isnonnegint(x):
    """Determine if x is a non-negative integer."""
    try:
        _ = operator.index(x)
    except TypeError:
        return False
    return x >= 0


def isposint(x):
    """Determine if x is a strictly positive integer."""
    return isnonnegint(x) and (x > 0)


def errorif(
    cond: Union[bool, jax.Array], err: Type[Exception] = ValueError, msg: str = ""
):
    """Raise an error if condition is met.

    Similar to assert but allows wider range of Error types, rather than
    just AssertionError.
    """
    if cond:
        raise err(msg)


def warnif(
    cond: Union[bool, jax.Array], err: Type[Warning] = UserWarning, msg: str = ""
):
    """Throw a warning if condition is met."""
    if cond:
        warnings.warn(msg, err)


def asarray_inexact(x: Num[Arrayish, "..."]) -> Inexact[Array, "..."]:
    """Convert to jax array with floating point dtype."""
    x = jnp.asarray(x)
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(x, jnp.array(1.0))
    return x.astype(dtype)


def wrap_jit(*args, **kwargs):
    """Wrap a function with jit with optional extra args.

    This is a helper to ensure docstrings and type hints are correctly propagated
    to the wrapped function, bc vscode seems to have issues with regular jitted funcs.
    """

    def wrapper(fun):
        foo = jax.jit(fun, *args, **kwargs)
        foo = functools.wraps(fun)(foo)
        return foo

    return wrapper


class _Indexable:
    """Helper object for building indexes for indexed update functions.

    This is a singleton object that overrides the ``__getitem__`` method
    to return the index it is passed.
    >>> Index[1:2, 3, None, ..., ::2]
    (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
    copied from jax.ops.index to work with either backend
    """

    __slots__ = ()

    def __getitem__(self, index):
        return index

    @staticmethod
    def get(stuff, axis, ndim):
        slices = [slice(None)] * ndim
        slices[axis] = stuff
        slices = tuple(slices)
        return slices


"""
Helper object for building indexes for indexed update functions.
This is a singleton object that overrides the ``__getitem__`` method
to return the index it is passed.
>>> Index[1:2, 3, None, ..., ::2]
(slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
copied from jax.ops.index to work with either backend
"""
Index = _Indexable()
