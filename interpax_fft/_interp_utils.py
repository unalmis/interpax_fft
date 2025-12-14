"""Basic utilities for interpolation."""

import jax.numpy as jnp


def bijection_from_disc(x, a, b):
    """[−1, 1] ∋ x ↦ y ∈ [a, b]."""
    return 0.5 * (b - a) * (x + 1) + a


def bijection_to_disc(x, a, b):
    """[a, b] ∋ x ↦ y ∈ [−1, 1]."""
    return 2 * (x - a) / (b - a) - 1


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
