"""General utilities."""

import functools
import operator
import warnings
from typing import Any, Type, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Num
from numpy.typing import ArrayLike

# jax.typing.ArrayLike and jaxtyping.ArrayLike don't include eg tuples,lists,iterables
# like np.ArrayLike. This combines all the usual array types
Arrayish = Union[Array, ArrayLike]


def isnonnegint(x: Any) -> bool:
    """Determine if x is a non-negative integer."""
    try:
        _ = operator.index(x)
    except TypeError:
        return False
    return x >= 0


def isposint(x: Any) -> bool:
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
    if x.weak_type:  # preserve weakly typed things like scalars
        return x
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


def setdefault(val, default, cond=None):
    """Return val if condition is met, otherwise default.

    If cond is None, then it checks if val is not None, returning val
    or default accordingly.
    """
    return val if cond or (cond is None and val is not None) else default


def bijection_from_disc(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    return 0.5 * (b - a) * (x + 1) + a


def bijection_to_disc(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    return 2 * (x - a) / (b - a) - 1


def atleast_nd(ndmin, ary):
    """Adds dimensions to front if necessary."""
    return jnp.array(ary, ndmin=ndmin) if jnp.ndim(ary) < ndmin else ary


def atleast_3d_mid(ary):
    """Like np.atleast_3d but if adds dim at axis 1 for 2d arrays."""
    ary = jnp.atleast_2d(ary)
    return ary[:, jnp.newaxis] if ary.ndim == 2 else ary


def atleast_2d_end(ary):
    """Like np.atleast_2d but if adds dim at axis 1 for 1d arrays."""
    ary = jnp.atleast_1d(ary)
    return ary[:, jnp.newaxis] if ary.ndim == 1 else ary


def flatten_mat(y, axes=2):
    """Flatten matrix to vector.

    Parameters
    ----------
    axes : int
        Number of trailing axes to flatten into last dimension.
        Default is two.

    Returns
    -------
    y : jnp.ndarray
        Shape (*y.shape[:-axes], -1).

    """
    return y.reshape(*y.shape[:-axes], -1)


def subtract_first(c, k):
    """Subtract ``k`` from first index of last axis of ``c``.

    Semantically same as ``return c.at[...,0].subtract(k)``,
    but allows dimension to increase.
    """
    if jnp.ndim(k) == 0:
        return c.at[..., 0].subtract(k)
    c_0 = c[..., 0] - k
    return jnp.concatenate(
        [
            c_0[..., jnp.newaxis],
            jnp.broadcast_to(c[..., 1:], (*c_0.shape, c.shape[-1] - 1)),
        ],
        axis=-1,
    )


def filter_distinct(r, sentinel, eps):
    """Set all but one of matching adjacent elements in ``r``  to ``sentinel``."""
    # eps needs to be low enough that close distinct roots do not get removed.
    # Otherwise, algorithms relying on continuity will fail.
    mask = jnp.isclose(jnp.diff(r, axis=-1, prepend=sentinel), 0, atol=eps)
    return jnp.where(mask, sentinel, r)


# TODO: replace the inner loop in orthax with this
def chebder(c, m=1, scl=1.0, axis=0, keepdims=False):
    """Same as orthax.chebder but fast enough to use in optimization loop."""
    assert m == 1
    c = jnp.flip(c.swapaxes(axis, 0), 0)

    N = c.shape[0]
    n = jnp.arange(N - 1, -1, -1).reshape((N,) + (1,) * (c.ndim - 1))
    w = (2 * scl) * n * c

    dc = jnp.flip(
        jnp.zeros(c.shape)
        .at[1::2]
        .set(jnp.cumsum(w[::2], 0)[: N // 2])
        .at[2::2]
        .set(jnp.cumsum(w[1::2], 0)[: (N - 1) // 2])
        .at[-1]
        .multiply(0.5),
        0,
    )
    if not keepdims:
        dc = dc[:-1]
    dc = dc.swapaxes(axis, 0)
    return dc


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
