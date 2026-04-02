"""Public utilities useful for interpolation, root finding, etc. in JAX."""

from functools import partial

import jax.numpy as jnp
from jax.numpy import flatnonzero, take

from ._utils_private import Index, bijection_from_disc


def cheb_from_dct(a, axis=-1):
    """Get discrete Chebyshev transform from discrete cosine transform.

    Parameters
    ----------
    a : jnp.ndarray
        Discrete cosine transform coefficients, e.g.
        ``a=dct(f,type=2,axis=axis,norm="forward")``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    cheb : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    return a.at[Index.get(0, axis, a.ndim)].divide(2)


def dct_from_cheb(cheb, axis=-1):
    """Get discrete cosine transform from discrete Chebyshev transform.

    Parameters
    ----------
    cheb : jnp.ndarray
        Discrete Chebyshev transform coefficients, e.g.``cheb_from_dct(a)``.
    axis : int
        Axis along which to transform.

    Returns
    -------
    a : jnp.ndarray
        Chebyshev coefficients along ``axis``.

    """
    return cheb.at[Index.get(0, axis, cheb.ndim)].multiply(2)


def cheb_pts(n, domain=(-1, 1), lobatto=False):
    """Get ``n`` Chebyshev points mapped to given domain.

    Warnings
    --------
    This is a common definition of the Chebyshev points (see Boyd, Chebyshev and
    Fourier Spectral Methods p. 498). These are the points demanded by discrete
    cosine transformations to interpolate Chebyshev series because the cosine
    basis for the DCT is defined on [0, π]. They differ in ordering from the
    points returned by ``numpy.polynomial.chebyshev.chebpts1`` and
    ``numpy.polynomial.chebyshev.chebpts2``.

    Parameters
    ----------
    n : int
        Number of points.
    domain : tuple[float]
        Domain for points.
    lobatto : bool
        Whether to return the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots for Chebyshev points.

    Returns
    -------
    pts : jnp.ndarray
        Shape (n, ).
        Chebyshev points mapped to given domain.

    """
    N = jnp.arange(n)
    if lobatto:
        y = jnp.cos(jnp.pi * N / (n - 1))
    else:
        y = jnp.cos(jnp.pi * (2 * N + 1) / (2 * n))
    return bijection_from_disc(y, domain[0], domain[-1])


def fourier_pts(n, domain=(0, 2 * jnp.pi)):
    """Get ``n`` Fourier points in domain."""
    return jnp.linspace(domain[0], domain[1], n, endpoint=False)


@partial(jnp.vectorize, signature="(m),(m)->(m)")
def epigraph_and(points, df_dy, /):
    """Set intersection of epigraph of function f with the given set of points.

    Useful to get subset of points where the straight line path between
    adjacent points resides in the epigraph of a continuous map ``f``.

    Parameters
    ----------
    points : jnp.ndarray
        Boolean array indicating which indices correspond to points in the set.
    df_dy : jnp.ndarray
        Shape ``points.shape``.
        Sign of ∂f/∂y (yᵢ) for f(yᵢ) = 0.

    Returns
    -------
    points : jnp.ndarray
        Boolean array indicating whether element is in epigraph of
        given function and satisfies the stated condition.

    """
    # The following comments are in regards to the article:
    # Spectrally accurate, reverse-mode differentiable bounce-averaging
    # algorithm and its applications.
    # Kaya E. Unalmis et al.

    # The pairs y1 and y2 are boundaries of an integral only if y1 <= y2. For the
    # to be over wells, it is required that the first intersect has a non-positive
    # derivative. Now, by continuity, df_dy[...,k]<=0 implies df_dy[...,k+1]>=0, so
    # there can be at most one inversion, and if it exists, it must be at the first
    # pair. To correct the inversion, it suffices to disqualify the first intersect
    # as a right boundary, except under an edge case of a series of inflection points.
    idx = flatnonzero(points, size=2, fill_value=-1)
    edge_case = (
        (df_dy[idx[0]] == 0)
        & (df_dy[idx[1]] < 0)
        & points[idx[0]]
        & points[idx[1]]
        # In theory, we need to keep propagating this edge case, e.g.
        # (df_dy[..., 1] < 0) | (
        #     (df_dy[..., 1] == 0) & (df_dy[..., 2] < 0)...
        # ).
        # At each step, the likelihood that an intersection has already been lost
        # due to floating point errors grows, so the real solution is to pick a less
        # degenerate pitch value - one that does not ride the global extrema of f.
    )
    return points.at[idx[0]].set(edge_case)


@partial(jnp.vectorize, signature="(m),(m)->(n)", excluded={"size", "fill_value"})
def take_mask(a, mask, /, *, size=-1, fill_value=None):
    """JIT compilable method to return ``a[mask][:size]`` padded by ``fill_value``.

    Parameters
    ----------
    a : jnp.ndarray
        The source array.
    mask : jnp.ndarray
        Boolean mask to index into ``a``. Should have same shape as ``a``.
    size : int
        Elements of ``a`` at the first size True indices of ``mask`` will be returned.
        Size is clipped to be <= ``mask.size``.
        A negative value will be interpreted as ``mask.size``.
    fill_value : Any
        When there are fewer nonzero elements in ``mask`` than ``size``, the remaining
        elements will be filled with ``fill_value``. Defaults to NaN for inexact types,
        the largest negative value for signed types, the largest positive value for
        unsigned types, and True for booleans.

    Returns
    -------
    result : jnp.ndarray
        Shape (size, ).

    """
    assert a.shape == mask.shape
    size = mask.size if (size is None or size < 0) else min(size, mask.size)
    idx = flatnonzero(mask, size=size, fill_value=mask.size)
    return take(
        a,
        idx,
        mode="fill",
        fill_value=fill_value,
        unique_indices=True,
        indices_are_sorted=True,
    )
