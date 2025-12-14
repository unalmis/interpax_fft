"""Tests for non-uniform MMT interpolation."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from jax.numpy.fft import rfft, rfft2
from jax.scipy.fft import dct, idct
from numpy.polynomial.chebyshev import (
    cheb2poly,
    chebinterpolate,
    chebpts1,
    chebpts2,
    chebval,
)

from interpax_fft import (
    DoubleChebyshevSeries,
    FourierChebyshevSeries,
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    fourier_pts,
    ifft_mmt,
    interp_dct,
    interp_rfft,
    interp_rfft2,
    irfft_mmt,
    rfft_to_trig,
    trig_vander,
)
from interpax_fft._interp_utils import bijection_to_disc

config.update("jax_enable_x64", True)


def identity(x):
    return x


def _c_1d(x):
    """Test function for 1D FFT."""
    return jnp.cos(7 * x) + jnp.sin(x) - 33.2


def _c_1d_nyquist_freq():
    return 7


def _c_2d(x, y):
    """Test function for 2D FFT."""
    x_freq, y_freq = 3, 5
    return (
        # something that's not separable
        jnp.cos(x_freq * x) * jnp.sin(2 * x + y)
        + jnp.sin(y_freq * y) * jnp.cos(x + 3 * y)
        - 33.2
        + jnp.cos(x)
        + jnp.cos(y)
    )


def _c_2d_nyquist_freq():
    x_freq, y_freq = 3, 5
    x_freq_nyquist = x_freq + 2
    y_freq_nyquist = y_freq + 3
    return x_freq_nyquist, y_freq_nyquist


def _f_non_periodic(z):
    return np.sin(np.sqrt(2) * z) * np.cos(1 / (2 + z)) * np.cos(z**2) * z


def _f_algebraic(z):
    return z**3 - 10 * z**6 - z - np.e + z**4


_test_inputs_1D = [
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (0, 2 * jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq() + 1, (-jnp.pi, jnp.pi)),
    (_c_1d, 2 * _c_1d_nyquist_freq(), (-jnp.pi, jnp.pi)),
    (lambda x: jnp.cos(7 * x), 2, (-jnp.pi / 7, jnp.pi / 7)),
    (lambda x: jnp.sin(7 * x), 3, (-jnp.pi / 7, jnp.pi / 7)),
]

_test_inputs_2D = [
    (
        _c_2d,
        2 * _c_2d_nyquist_freq()[0] + 1,
        2 * _c_2d_nyquist_freq()[1] + 1,
        (0, 2 * jnp.pi),
        (0, 2 * jnp.pi),
    ),
    (
        _c_2d,
        2 * _c_2d_nyquist_freq()[0] + 1,
        2 * _c_2d_nyquist_freq()[1] + 1,
        (-jnp.pi / 3, 5 * jnp.pi / 3),
        (jnp.pi, 3 * jnp.pi),
    ),
    (
        lambda x, y: jnp.cos(30 * x) + jnp.sin(y) ** 2 + 1,
        2 * 30 // 30 + 1,
        2 * 2 + 1,
        (0, 2 * jnp.pi / 30),
        (jnp.pi, 3 * jnp.pi),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("M", [1, 8, 9])
def test_fft_shift(M):
    """Test frequency shifting."""
    a = np.fft.rfftfreq(M, 1 / M)
    np.testing.assert_allclose(a, np.arange(M // 2 + 1))
    b = np.fft.fftfreq(a.size, 1 / a.size) + a.size // 2
    np.testing.assert_allclose(np.fft.ifftshift(a), b)


@pytest.mark.unit
@pytest.mark.parametrize(
    "func, n, domain, imag_undersampled",
    [
        (*_test_inputs_1D[0], False),
        (*_test_inputs_1D[1], True),
        (*_test_inputs_1D[2], False),
        (*_test_inputs_1D[3], True),
        (*_test_inputs_1D[4], True),
        (*_test_inputs_1D[5], False),
    ],
)
def test_non_uniform_FFT(func, n, domain, imag_undersampled):
    """Test non-uniform FFT interpolation."""
    x = np.linspace(domain[0], domain[1], n, endpoint=False)
    c = func(x)
    xq = np.array([7.34, 1.10134, 2.28])

    f = np.fft.fft(c, norm="forward")
    np.testing.assert_allclose(f[0].imag, 0, atol=1e-12)
    if n % 2 == 0:
        np.testing.assert_allclose(f[n // 2].imag, 0, atol=1e-12)

    r = ifft_mmt(xq, f, domain)
    np.testing.assert_allclose(r.real if imag_undersampled else r, func(xq))


@pytest.mark.unit
@pytest.mark.parametrize("func, n, domain", _test_inputs_1D)
def test_non_uniform_real_MMT(func, n, domain):
    """Test non-uniform real MMT interpolation."""
    x = np.linspace(domain[0], domain[1], n, endpoint=False)
    c = func(x)
    xq = np.array([7.34, 1.10134, 2.28])

    np.testing.assert_allclose(interp_rfft(xq, c, domain), func(xq))
    vand = trig_vander(xq, c.shape[-1], domain)
    coef = rfft_to_trig(rfft(c, norm="forward"), c.shape[-1])
    np.testing.assert_allclose((vand * coef).sum(-1), func(xq))


@pytest.mark.unit
@pytest.mark.parametrize("func, m, n, domain_x, domain_y", _test_inputs_2D)
def test_non_uniform_real_MMT_2D(func, m, n, domain_x, domain_y):
    """Test non-uniform real MMT 2D interpolation."""
    x = np.linspace(domain_x[0], domain_x[1], m, endpoint=False)
    y = np.linspace(domain_y[0], domain_y[1], n, endpoint=False)
    x, y = map(np.ravel, tuple(np.meshgrid(x, y, indexing="ij")))
    c = func(x, y).reshape(m, n)
    xq = np.array([7.34, 1.10134, 2.28, 1e3 * np.e])
    yq = np.array([1.1, 3.78432, 8.542, 0])

    v = func(xq, yq)
    np.testing.assert_allclose(interp_rfft2(xq, yq, c, domain_x, domain_y, (-2, -1)), v)
    np.testing.assert_allclose(interp_rfft2(xq, yq, c, domain_x, domain_y, (-1, -2)), v)
    np.testing.assert_allclose(
        interp_rfft2(yq, xq, c.T, domain_y, domain_x, (-2, -1)), v
    )
    np.testing.assert_allclose(
        interp_rfft2(yq, xq, c.T, domain_y, domain_x, (-1, -2)), v
    )
    np.testing.assert_allclose(
        irfft_mmt(
            yq,
            ifft_mmt(xq[:, None], rfft2(c, norm="forward"), domain_x, -2),
            n,
            domain_y,
        ),
        v,
    )


@pytest.mark.unit
@pytest.mark.parametrize("N", [2, 6, 7])
def test_cheb_pts(N):
    """Test we use Chebyshev points compatible with DCT."""
    np.testing.assert_allclose(cheb_pts(N), chebpts1(N)[::-1], atol=1e-15)
    np.testing.assert_allclose(
        cheb_pts(N, domain=(-np.pi, np.pi), lobatto=True),
        np.pi * chebpts2(N)[::-1],
        atol=1e-15,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "f, M, lobatto",
    [
        # Identity map known for bad Gibbs; if discrete Chebyshev transform
        # implemented correctly then won't see Gibbs.
        (identity, 2, False),
        (identity, 3, False),
        (identity, 3, True),
        (identity, 4, True),
    ],
)
def test_dct(f, M, lobatto):
    """Test discrete cosine transform interpolation.

    Parameters
    ----------
    f : callable
        Function to test.
    M : int
        Fourier spectral resolution.
    lobatto : bool
        Whether ``f`` should be sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        or interior roots grid for Chebyshev points.

    """
    # Need to test interpolation due to issues like
    # https://github.com/google/jax/issues/22466
    # https://github.com/jax-ml/jax/issues/23827
    # https://github.com/google/jax/issues/23895
    # https://github.com/jax-ml/jax/issues/31836
    # Unresolved:
    # https://github.com/jax-ml/jax/issues/29426
    # https://github.com/jax-ml/jax/issues/29325
    from scipy.fft import dct as sdct
    from scipy.fft import dctn as sdctn
    from scipy.fft import idct as sidct

    domain = (0, 2 * np.pi)
    m = cheb_pts(M, domain, lobatto)
    n = cheb_pts(m.size * 10, domain, lobatto)
    norm = (n.size - lobatto) / (m.size - lobatto)

    dct_type = 2 - lobatto
    fq_1 = np.sqrt(norm) * sidct(
        sdct(f(m), type=dct_type, norm="ortho", orthogonalize=False),
        type=dct_type,
        n=n.size,
        norm="ortho",
        orthogonalize=False,
    )
    if lobatto:
        # JAX has yet to implement type 1 DCT.
        fq_2 = norm * sidct(sdct(f(m), type=dct_type), n=n.size, type=dct_type)
    else:
        fq_2 = norm * idct(dct(f(m), type=dct_type), n=n.size, type=dct_type)
    np.testing.assert_allclose(fq_1, f(n), atol=1e-14)
    np.testing.assert_allclose(fq_2, f(n), atol=1e-14)

    if not lobatto:
        g = f(m)[:, None] * _f_algebraic(cheb_pts(7))
        cheb = DoubleChebyshevSeries(g)
        np.testing.assert_allclose(
            cheb._c,
            sdctn(g) / g.size,
            atol=1e-14,
            err_msg="Scipy and JAX disagree.",
        )
        truth = f(n)[:, None] * _f_algebraic(cheb_pts(8))
        np.testing.assert_allclose(cheb.evaluate(n.size, 8), truth)


@pytest.mark.unit
@pytest.mark.parametrize(
    "f, M",
    [(_f_non_periodic, 5), (_f_non_periodic, 6), (_f_algebraic, 7)],
)
def test_interp_dct(f, M):
    """Test non-uniform DCT interpolation."""
    c0 = chebinterpolate(f, M - 1)
    assert not np.allclose(
        c0,
        cheb_from_dct(dct(f(chebpts1(M)), 2)) / M,
    ), (
        "Interpolation should fail because cosine basis is in wrong domain, "
        "yet the supplied test function was interpolated fine using this wrong "
        "domain. Pick a better test function."
    )

    z = cheb_pts(M)
    fz = f(z)
    np.testing.assert_allclose(c0, cheb_from_dct(dct(fz, 2) / M), atol=1e-13)
    if np.allclose(_f_algebraic(z), fz):  # Should reconstruct exactly.
        np.testing.assert_allclose(
            cheb2poly(c0),
            np.array([-np.e, -1, 0, 1, 1, 0, -10]),
            atol=1e-13,
        )

    xq = np.arange(10 * 3 * 2).reshape(10, 3, 2)
    xq = bijection_to_disc(xq, 0, xq.size)
    fq = chebval(xq, c0, tensor=False)
    np.testing.assert_allclose(fq, interp_dct(xq, fz), atol=1e-13)


@pytest.mark.parametrize(
    "func, X, Y",
    [(_c_2d, 2 * _c_2d_nyquist_freq()[0] + 1, 2 * _c_2d_nyquist_freq()[1] + 1)],
)
def test_double_chebyshev(func, X, Y):
    """Tests for coverage of DoubleChebyshev series."""
    _, x, y = DoubleChebyshevSeries.nodes(X, Y, L=1).T
    f = func(x, y).reshape(X, Y)
    series = DoubleChebyshevSeries(f)
    np.testing.assert_allclose(series.evaluate(X, Y), f)

    c = series.compute_cheb(cheb_pts(X))
    c = dct_from_cheb(c)
    c = dct(c, axis=-2) / X
    assert c.shape == series._c.shape
    np.testing.assert_allclose(c, series._c, atol=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize(
    "func, X, Y",
    [(_c_2d, 2 * _c_2d_nyquist_freq()[0] + 1, 2 * _c_2d_nyquist_freq()[1] + 1)],
)
def test_fourier_chebyshev(func, X, Y):
    """Tests for coverage of FourierChebyshev series."""
    x, y = FourierChebyshevSeries.nodes(X, Y).T
    f = func(x, y).reshape(X, Y)
    series = FourierChebyshevSeries(f)
    np.testing.assert_allclose(series.evaluate(X, Y), f)

    c = series.compute_cheb(fourier_pts(X))
    c = dct_from_cheb(c)
    c = rfft(c, axis=-2, norm="forward")
    assert c.shape == series._c.shape
    np.testing.assert_allclose(c, series._c, atol=1e-14)

    assert np.isfinite(series.harmonics()).all()
