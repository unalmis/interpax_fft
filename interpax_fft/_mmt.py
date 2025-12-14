"""Non-uniform MMT interpolation."""

import jax.numpy as jnp
import numpy as np
from jax.numpy.fft import rfft, rfft2
from jax.scipy.fft import dct

from ._utils import Index, errorif


def interp_rfft(x, f, domain=(0, 2 * jnp.pi), axis=-1):
    """Interpolate real-valued ``f`` to ``x`` with FFT and MMT.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(f.shape,axis)``.
    f : jnp.ndarray
        Real function values on uniform grid over an open period to interpolate.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    return irfft_mmt(x, rfft(f, axis=axis, norm="forward"), f.shape[axis], domain, axis)


def irfft_mmt(x, a, n, domain=(0, 2 * jnp.pi), axis=-1):
    """Evaluate Fourier coefficients ``a`` at ``x``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,axis=axis,norm="forward")``.
    n : int
        Spectral resolution of ``a``.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    modes = jnp.fft.rfftfreq(n, (domain[1] - domain[0]) / (2 * jnp.pi * n))
    i = (0, -1) if (n % 2 == 0) else 0
    a = jnp.moveaxis(a, axis, -1).at[..., i].divide(2) * 2
    vander = jnp.exp(1j * modes * (x - domain[0])[..., None])
    return jnp.einsum("...m, ...m", vander, a).real


def irfft_mmt_pos(x, a, n, domain=(0, 2 * jnp.pi), modes=None):
    """Evaluate Fourier coefficients ``a`` at ``x``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(a.shape,-1)``.
    a : jnp.ndarray
        Fourier coefficients of non-negative frequencies.
    n : int
        Spectral resolution of ``a``.
    domain : tuple[float]
        Domain over which samples were taken.
    modes : jnp.ndarray
        Optional frequency modes.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    if modes is None:
        modes = jnp.fft.rfftfreq(n, (domain[1] - domain[0]) / (2 * jnp.pi * n))
    vander = jnp.exp(1j * modes * (x - domain[0])[..., None])
    return jnp.einsum("...m, ...m", vander, a).real


def ifft_mmt(x, a, domain=(0, 2 * jnp.pi), axis=-1, *, vander=None, modes=None):
    """Evaluate Fourier coefficients ``a`` at ``x``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with arrays of shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Fourier coefficients ``a=fft(f,axis=axis,norm="forward")``.
    domain : tuple[float]
        Domain over which samples were taken.
    axis : int
        Axis along which to transform.
    vander : jnp.ndarray
        Precomputed transform matrix.
        If given returns ``(vander*a).sum(axis)``.
    modes : jnp.ndarray
        Precomputed modes.

    Returns
    -------
    fq : jnp.ndarray
        Function value at query points.

    """
    if vander is None:
        if modes is None:
            n = a.shape[axis]
            modes = jnp.fft.fftfreq(n, (domain[1] - domain[0]) / (2 * jnp.pi * n))
        vander = jnp.exp(1j * modes * (x - domain[0])[..., None])
        a = jnp.moveaxis(a, axis, -1)
        axis = -1
    return (vander * a).sum(axis)


def interp_rfft2(
    x0, x1, f, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Interpolate real-valued ``f`` to coordinates ``(x0,x1)`` with FFT and MMT.

    Parameters
    ----------
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the function values ``f``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the function values ``f``.
    f : jnp.ndarray
        Shape (..., f.shape[-2], f.shape[-1]).
        Real function values on uniform tensor-product grid over an open period.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.
        The real transform is done along ``axes[1]``, so it will be more
        efficient for that to denote the smaller size axis in ``axes``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    i = (0, -1) if (f.shape[axes[1]] % 2 == 0) else 0
    a = rfft2(f, axes=axes, norm="forward")
    a = jnp.moveaxis(a, axes, (-2, -1)).at[..., i].divide(2) * 2
    n0, n1 = sorted(axes)
    return irfft2_mmt_pos(
        x0,
        x1,
        a,
        f.shape[n0],
        f.shape[n1],
        domain0,
        domain1,
        axes,
    )


def irfft2_mmt_pos(
    x0, x1, a, n0, n1, domain0=(0, 2 * jnp.pi), domain1=(0, 2 * jnp.pi), axes=(-2, -1)
):
    """Evaluate Fourier coefficients ``a`` at coordinates ``(x0,x1)``.

    Uses matrix multiplication transform.

    Parameters
    ----------
    x0 : jnp.ndarray
        Real query points of coordinate in ``domain0`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``min(axes)`` of the Fourier coefficients ``a``.
    x1 : jnp.ndarray
        Real query points of coordinate in ``domain1`` where interpolation is desired.
        Shape must broadcast with shape ``np.delete(a.shape,axes)``.
        The coordinates stored here must be the same coordinate enumerated
        across axis ``max(axes)`` of the Fourier coefficients ``a``.
    a : jnp.ndarray
        Shape (..., a.shape[-2], a.shape[-1]).
        Fourier coefficients.
        ``f=rfft2(f,axes=axes,norm="forward")``
        ``a=jnp.moveaxis(f,axes,(-2,-1)).at[...,i].divide(2)*2``.
    n0 : int
        Spectral resolution of ``a`` for ``domain0``.
    n1 : int
        Spectral resolution of ``a`` for ``domain1``.
    domain0 : tuple[float]
        Domain of coordinate specified by x₀ over which samples were taken.
    domain1 : tuple[float]
        Domain of coordinate specified by x₁ over which samples were taken.
    axes : tuple[int]
        Axes along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    x = (x0, x1)
    n = (n0, n1)
    d = (domain0, domain1)
    f, r = np.argsort(axes)
    vf, vr = rfft2_modes(n[f], n[r], d[f], d[r])
    vf = jnp.exp(1j * vf * (x[f] - d[f][0])[..., None])
    vr = jnp.exp(1j * vr * (x[r] - d[r][0])[..., None])

    if n[f] > n[r]:
        return jnp.einsum("...r, ...r", vr, jnp.einsum("...f, ...fr", vf, a)).real
    else:
        return jnp.einsum("...f, ...f", vf, jnp.einsum("...r, ...fr", vr, a)).real


def rfft2_vander(
    x_fft,
    x_rfft,
    modes_fft,
    modes_rfft,
    x_fft0=0,
    x_rfft0=0,
    inverse_idx_fft=None,
    inverse_idx_rfft=None,
):
    """Return Vandermonde matrix for complex Fourier modes.

    Parameters
    ----------
    x_fft : jnp.ndarray
        Real query points of coordinate in ``domain_fft`` where interpolation is
        desired.
    x_rfft : jnp.ndarray
        Real query points of coordinate in ``domain_rfft`` where interpolation is
        desired.
    modes_fft : jnp.ndarray
        FFT Fourier modes.
    modes_rfft : jnp.ndarray
        Real FFT Fourier modes.
    x_fft0 : float
        Left boundary of domain of coordinate specified by ``x_fft`` over which
        samples were taken.
    x_rfft0 : float
        Left boundary of domain of coordinate specified by ``x_rfft`` over which
        samples were taken.
    inverse_idx_fft : jnp.ndarray
        Optional. Inverse idx along axis 0 to ensure query points broadcast.
    inverse_idx_rfft : jnp.ndarray
        Optional. Inverse idx along axis 0 to ensure query points broadcast.

    Returns
    -------
    vander : jnp.ndarray
        Shape (..., modes_fft.size, modes_rfft.size).
        Vandermonde matrix to evaluate complex Fourier series.

    """
    vf = jnp.exp(1j * modes_fft * (x_fft - x_fft0)[..., None])
    vr = jnp.exp(1j * modes_rfft * (x_rfft - x_rfft0)[..., None])
    if inverse_idx_fft is not None:
        vf = vf[inverse_idx_fft]
    if inverse_idx_rfft is not None:
        vr = vr[inverse_idx_rfft]
    return vf[..., None] * vr[..., None, :]


def rfft2_modes(n_fft, n_rfft, domain_fft=(0, 2 * jnp.pi), domain_rfft=(0, 2 * jnp.pi)):
    """Modes for complex exponential basis for real Fourier transform.

    Parameters
    ----------
    n_fft : int
        Spectral resolution for ``domain_fft``.
    n_rfft : int
        Spectral resolution for ``domain_rfft``.
    domain_fft : tuple[float]
        Domain of coordinate over which samples are taken.
    domain_rfft : tuple[float]
        Domain of coordinate over which samples are taken.

    Returns
    -------
    modes_fft : jnp.ndarray
        Shape (n_fft, ).
        FFT Fourier modes.
    modes_rfft : jnp.ndarray
        Shape (n_rfft // 2 + 1, ).
        Real FFT Fourier modes.

    """
    modes_fft = jnp.fft.fftfreq(
        n_fft, (domain_fft[1] - domain_fft[0]) / (2 * jnp.pi * n_fft)
    )
    modes_rfft = jnp.fft.rfftfreq(
        n_rfft, (domain_rfft[1] - domain_rfft[0]) / (2 * jnp.pi * n_rfft)
    )
    return modes_fft, modes_rfft


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


def interp_dct(x, f, lobatto=False, axis=-1):
    """Interpolate ``f`` to ``x`` with discrete Chebyshev transform.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with shape ``np.delete(f.shape,axis)``.
    f : jnp.ndarray
        Real function values on Chebyshev points to interpolate.
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        or interior roots grid for Chebyshev points.
    axis : int
        Axis along which to transform.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
    return idct_mmt(
        x,
        cheb_from_dct(dct(f, type=2 - lobatto, axis=axis), axis)
        / (f.shape[axis] - lobatto),
        axis,
    )


def idct_mmt(x, a, axis=-1, vander=None):
    """Evaluate Chebyshev coefficients ``a`` at ``x`` ∈ [-1, 1].

    Uses matrix multiplication transform, which was benchmarked
    to be significantly faster than Clenshaw recursion.

    Parameters
    ----------
    x : jnp.ndarray
        Real query points where interpolation is desired.
        Shape of ``x`` must broadcast with shape ``np.delete(a.shape,axis)``.
    a : jnp.ndarray
        Discrete Chebyshev transform coefficients.
    axis : int
        Axis along which to transform.
    vander : jnp.ndarray
        Precomputed transform matrix.
        If given returns ``(vander*a).sum(-1)``.

    Returns
    -------
    fq : jnp.ndarray
        Real function value at query points.

    """
    if vander is None:
        vander = jnp.cos(jnp.arange(a.shape[axis]) * jnp.arccos(x)[..., None])
        a = jnp.moveaxis(a, axis, -1)

    # Same as (in infinite precision) chebval(x,jnp.moveaxis(a,axis,0),tensor=False),
    # except faster on CPU and GPU.
    return jnp.einsum("...m, ...m", vander, a)


def rfft_to_trig(a, n, axis=-1):
    """Spectral coefficients of the Nyquist trigonometric interpolant.

    Parameters
    ----------
    a : jnp.ndarray
        Fourier coefficients ``a=rfft(f,norm="forward",axis=axis)``.
    n : int
        Spectral resolution of ``a``.
    axis : int
        Axis along which coefficients are stored.

    Returns
    -------
    h : jnp.ndarray
        Nyquist trigonometric interpolant coefficients.

        Coefficients are ordered along ``axis`` of size ``n`` to match
        Vandermonde matrix with order
        [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)].
        When ``n`` is even the sin(k𝐱) coefficient is zero and is excluded.

    """
    is_even = (n % 2) == 0
    an = -2 * jnp.flip(
        jnp.take(
            a.imag,
            jnp.arange(1, a.shape[axis] - is_even),
            axis,
            unique_indices=True,
            indices_are_sorted=True,
        ),
        axis=axis,
    )
    i = (0, -1) if is_even else 0
    bn = a.real.at[Index.get(i, axis, a.ndim)].divide(2) * 2
    h = jnp.concatenate([an, bn], axis=axis)
    assert h.shape[axis] == n
    return h


def trig_vander(x, n, domain=(0, 2 * jnp.pi)):
    """Nyquist trigonometric interpolant basis evaluated at ``x``.

    Parameters
    ----------
    x : jnp.ndarray
        Points at which to evaluate Vandermonde matrix.
    n : int
        Spectral resolution.
    domain : tuple[float]
        Domain over which samples will be taken.
        This domain should span an open period of the function to interpolate.

    Returns
    -------
    vander : jnp.ndarray
        Shape (*x.shape, n).
        Vandermonde matrix of degree ``n-1`` and sample points ``x``.
        Last axis ordered as [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)].
        When ``n`` is even the sin(k𝐱) basis function is excluded.

    """
    is_even = (n % 2) == 0
    n_rfft = jnp.fft.rfftfreq(n, d=(domain[-1] - domain[0]) / (2 * jnp.pi * n))
    nx = n_rfft * (x - domain[0])[..., None]
    return jnp.concatenate(
        [jnp.sin(nx[..., n_rfft.size - is_even - 1 : 0 : -1]), jnp.cos(nx)], axis=-1
    )
