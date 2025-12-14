"""Fast transformable series."""

import jax
import jax.numpy as jnp
from equinox import Module
from jax.numpy.fft import irfft, rfft
from jax.scipy.fft import dct, dctn, idct, idctn

from ._interp_utils import bijection_to_disc, cheb_pts, fourier_pts
from ._mmt import cheb_from_dct, idct_mmt, irfft_mmt, rfft_to_trig
from ._utils import errorif, isposint, warnif


class DoubleChebyshevSeries(Module):
    """Real-valued 2D Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ Tₘ(x) Tₙ(y)
    where Tₘ are Chebyshev polynomials on [xₘᵢₙ, xₘₐₓ]
    and Tₙ are Chebyshev polynomials on [yₘᵢₙ, yₘₐₓ].

    Notes
    -----
    Performance may improve if ``X`` and ``Y`` are powers of two.

    Parameters
    ----------
    f : jnp.ndarray
        Shape (..., X, Y).
        Samples of real function on the ``ChebyshevSeries.nodes`` grid.
    domain_x : tuple[float]
        Domain for x coordinates. Default is [-1, 1].
    domain_y : tuple[float]
        Domain for y coordinates. Default is [-1, 1].
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots grid for Chebyshev points.
    truncate_x: int
        Index at which to truncate the Chebyshev series in x coordinate.
        This will remove aliasing error at the shortest wavelengths where the signal
        to noise ratio is lowest. The default value is zero which is interpreted as
        no truncation.
    truncate_y: int
        Index at which to truncate the Chebyshev series in y coordinate.
        This will remove aliasing error at the shortest wavelengths where the signal
        to noise ratio is lowest. The default value is zero which is interpreted as
        no truncation.

    Attributes
    ----------
    X : int
        Chebyshev spectral resolution in x coordinate.
    Y : int
        Chebyshev spectral resolution in y coordinate.

    """

    X: int
    Y: int
    domain_x: tuple[float]
    domain_y: tuple[float]
    lobatto: bool
    _c: jax.Array

    def __init__(
        self,
        f,
        domain_x=(-1, 1),
        domain_y=(-1, 1),
        lobatto=False,
        truncate_x=0,
        truncate_y=0,
    ):
        """Interpolate Chebyshev-Chebyshev series to ``f``."""
        errorif(domain_x[0] > domain_x[-1], msg="Got inverted x domain.")
        errorif(domain_y[0] > domain_y[-1], msg="Got inverted y domain.")
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.lobatto = lobatto
        X = f.shape[-2]
        Y = f.shape[-1]
        self.X = truncate_x if (0 < truncate_x < X) else X
        self.Y = truncate_y if (0 < truncate_y < Y) else Y
        self._c = dctn(f, type=2 - lobatto, axes=(-2, -1))[..., : self.X, : self.Y] / (
            (X - lobatto) * (Y - lobatto)
        )

    @staticmethod
    def nodes(X, Y, L=None, domain_x=(-1, 1), domain_y=(-1, 1), lobatto=False):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.
        L : int or jnp.ndarray
            Optional, resolution in radial direction of domain [0, 1].
            May also be an array of coordinates values. If given, then the
            returned ``coords`` is a 3D tensor-product with shape (L * X * Y, 3).
        domain_x : tuple[float]
            Domain for x coordinates. Default is [-1, 1].
        domain_y : tuple[float]
            Domain for y coordinates. Default is [-1, 1].
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            instead of the interior roots grid for Chebyshev points.

        Returns
        -------
        coords : jnp.ndarray
            Shape (X * Y, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        return _coords(
            cheb_pts(X, domain_x, lobatto), cheb_pts(Y, domain_y, lobatto), L
        )

    def evaluate(self, X, Y):
        """Evaluate Chebyshev series on tensor-product grid.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Chebyshev series evaluated at
            ``ChebyshevSeries.nodes(X,Y,L,self.domain_x,self.domain_y,self.lobatto)``.

        """
        warnif(
            X < self.X,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got X = {X} < {self.X} = self.X.",
        )
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        BUG = jax.__version_info__ <= (0, 7, 2)
        errorif(
            self._c.ndim > 2 and BUG,
            msg="https://github.com/jax-ml/jax/issues/31836",
        )
        axes = None if BUG else (-2, -1)
        return (
            idctn(self._c, type=2 - self.lobatto, s=(X, Y), axes=axes)
            * (X - self.lobatto)
            * (Y - self.lobatto)
        )

    def compute_cheb(self, x):
        """Evaluate at coordinate ``x`` to get set of 1D Chebyshev series in ``y``.

        Parameters
        ----------
        x : jnp.ndarray
            Points to evaluate Chebyshev series.

        Returns
        -------
        cheb : jnp.ndarray
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        x = bijection_to_disc(x, self.domain_x[0], self.domain_x[-1])
        # Add axis to broadcast against Chebyshev series in y.
        x = jnp.atleast_1d(x)[..., None]
        cheb = cheb_from_dct(cheb_from_dct(self._c, -2), -1)
        # Add axis to broadcast against multiple x values.
        cheb = idct_mmt(x, cheb[..., None, :, :], -2)
        assert cheb.shape[-2:] == (x.shape[-2], self.Y)
        return cheb


class FourierChebyshevSeries(Module):
    """Real-valued Fourier-Chebyshev series.

    f(x, y) = ∑ₘₙ aₘₙ ψₘ(x) Tₙ(y)
    where ψₘ are trigonometric polynomials on [0, 2π]
    and Tₙ are Chebyshev polynomials on [yₘᵢₙ, yₘₐₓ].

    Notes
    -----
    Performance may improve if ``X`` and ``Y`` are powers of two.


    Parameters
    ----------
    f : jnp.ndarray
        Shape (..., X, Y).
        Samples of real function on the ``FourierChebyshevSeries.nodes`` grid.
    domain : tuple[float]
        Domain for y coordinates. Default is [-1, 1].
    lobatto : bool
        Whether ``f`` was sampled on the Gauss-Lobatto (extrema-plus-endpoint)
        instead of the interior roots grid for Chebyshev points.
    truncate : int
        Index at which to truncate the Chebyshev series.
        This will remove aliasing error at the shortest wavelengths where the signal
        to noise ratio is lowest. The default value is zero which is interpreted as
        no truncation.

    Attributes
    ----------
    X : int
        Fourier spectral resolution.
    Y : int
        Chebyshev spectral resolution.

    """

    X: int
    Y: int
    domain: tuple[float]
    lobatto: bool
    _c: jax.Array

    def __init__(self, f, domain=(-1, 1), lobatto=False, truncate=0):
        """Interpolate Fourier-Chebyshev series to ``f``."""
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        errorif(lobatto, NotImplementedError, "JAX has not implemented type 1 DCT.")
        self.domain = domain
        self.lobatto = lobatto
        self.X = f.shape[-2]
        Y = f.shape[-1]
        self.Y = truncate if (0 < truncate < Y) else Y
        self._c = rfft(
            dct(f, type=2 - lobatto, axis=-1)[..., : self.Y] / (Y - lobatto),
            axis=-2,
            norm="forward",
        )

    @staticmethod
    def nodes(X, Y, L=None, domain=(-1, 1), lobatto=False):
        """Tensor product grid of optimal collocation nodes for this basis.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.
        L : int or jnp.ndarray
            Optional, resolution in radial direction of domain [0, 1].
            May also be an array of coordinates values. If given, then the
            returned ``coords`` is a 3D tensor-product with shape (L * X * Y, 3).
        domain : tuple[float]
            Domain for y coordinates. Default is [-1, 1].
        lobatto : bool
            Whether to use the Gauss-Lobatto (Extrema-plus-Endpoint)
            instead of the interior roots grid for Chebyshev points.

        Returns
        -------
        coords : jnp.ndarray
            Shape (X * Y, 2).
            Grid of (x, y) points for optimal interpolation.

        """
        return _coords(fourier_pts(X), cheb_pts(Y, domain, lobatto), L)

    def evaluate(self, X, Y):
        """Evaluate Fourier-Chebyshev series on tensor-product grid.

        Parameters
        ----------
        X : int
            Grid resolution in x direction. Preferably power of 2.
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Fourier-Chebyshev series evaluated at
            ``FourierChebyshevSeries.nodes(X,Y,L,self.domain,self.lobatto)``.

        """
        # TODO: Preserve spectrum using same logic as _fft.py
        warnif(
            X < self.X,
            msg="Frequency spectrum of FFT interpolation will be truncated because "
            "the grid resolution is less than the Fourier resolution.\n"
            f"Got X = {X} < {self.X} = self.X.",
        )
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        return idct(
            irfft(self._c, n=X, axis=-2, norm="forward"),
            type=2 - self.lobatto,
            n=Y,
            axis=-1,
        ) * (Y - self.lobatto)

    def compute_cheb(self, x):
        """Evaluate at coordinate ``x`` to get set of 1D Chebyshev series in ``y``.

        Parameters
        ----------
        x : jnp.ndarray
            Points to evaluate Fourier series.

        Returns
        -------
        cheb : jnp.ndarray
            Chebyshev coefficients αₙ(x=``x``) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y).

        """
        # Add axis to broadcast against Chebyshev series in y.
        x = jnp.atleast_1d(x)[..., None]
        # Add axis to broadcast against multiple x values.
        cheb = cheb_from_dct(irfft_mmt(x, self._c[..., None, :, :], self.X, axis=-2))
        assert cheb.shape[-2:] == (x.shape[-2], self.Y)
        return cheb

    def harmonics(self):
        """Real spectral coefficients aₘₙ of the interpolating polynomial.

        The order of the returned coefficient array
        matches the Vandermonde matrix formed by an outer
        product of Fourier and Chebyshev matrices with order
        [sin(k𝐱), ..., sin(𝐱), 1, cos(𝐱), ..., cos(k𝐱)]
        ⊗ [T₀(𝐲), T₁(𝐲), ..., Tₙ(𝐲)]

        When ``self.X`` is even the sin(k𝐱) coefficient is zero and is excluded.

        Returns
        -------
        a_mn : jnp.ndarray
            Shape (..., X, Y).
            Real valued spectral coefficients for Fourier-Chebyshev series.

        """
        return rfft_to_trig(cheb_from_dct(self._c), self.X, axis=-2)


def _coords(x, y, L):
    if L is None:
        coords = (x, y)
    else:
        if isposint(L):
            L = jnp.flipud(jnp.linspace(1, 0, L, endpoint=False))
        coords = (jnp.atleast_1d(L), x, y)
    coords = tuple(map(jnp.ravel, jnp.meshgrid(*coords, indexing="ij")))
    return jnp.column_stack(coords)
