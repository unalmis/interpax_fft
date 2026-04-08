"""Function approximation with series expansions."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module
from jax.numpy.fft import irfft, rfft
from jax.scipy.fft import dct, dctn, idct, idctn

from ._mmt import idct_mmt, irfft_mmt, rfft_to_trig
from ._utils_private import (
    atleast_2d_end,
    atleast_3d_mid,
    atleast_nd,
    bijection_from_disc,
    bijection_to_disc,
    errorif,
    filter_distinct,
    flatten_mat,
    isposint,
    setdefault,
    subtract_first,
    warnif,
)
from ._utils_public import (
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    epigraph_and,
    fourier_pts,
    take_mask,
)

try:
    from orthax.chebyshev import chebroots

    _chebroots_vec = jnp.vectorize(chebroots, signature="(m)->(n)")
except ImportError:
    # not an installation requirement
    from numpy.polynomial.chebyshev import chebroots

    _chebroots_vec = np.vectorize(chebroots, signature="(m)->(n)")

try:
    from matplotlib import pyplot as plt
except ImportError:
    # not an installation requirement
    pass

from packaging import version

_DCT_BUG = version.parse(jax.__version__) <= version.parse("0.7.2")


def _coords(x, y, L):
    if L is None:
        coords = (x, y)
    else:
        if isposint(L):
            L = jnp.flipud(jnp.linspace(1, 0, L, endpoint=False))
        coords = (jnp.atleast_1d(L), x, y)
    coords = tuple(map(jnp.ravel, jnp.meshgrid(*coords, indexing="ij")))
    return jnp.column_stack(coords)


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
        errorif(
            _DCT_BUG,
            msg="Your version of JAX has bugs.\n"
            "https://github.com/jax-ml/jax/issues/31836.\n"
            "https://github.com/jax-ml/jax/pull/34123.",
        )
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
        return (
            idctn(self._c, type=2 - self.lobatto, s=(X, Y), axes=(-2, -1))
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


@partial(jax.checkpoint, prevent_cse=False)
def _gather_reduce(y, cheb, x_idx):
    """Gather then reduce with checkpointing.

    Checkpointing this makes it faster and reduces memory.
    On many architectures, the cost of allocating contiguous blocks of memory
    exceeds the cost of flops. By checkpointing, we avoid that in favor of
    recomputing derivatives in the backward pass.

    On JAX version 0.7.2, enabling CSE did not increase memory.
    """
    Y = cheb.shape[-1]
    return jnp.einsum(
        "...m, ...m",
        jnp.cos(jnp.arange(Y) * y[..., None]),
        jnp.take_along_axis(cheb, x_idx[..., None], axis=-2),
    )


@partial(jax.checkpoint, prevent_cse=False)
def _loop(y, cheb, x_idx):
    """Memory efficient Clenshaw recursion.

    Checkpointing on a CPU observed a minor reduction in memory usage while not
    affecting speed. With/without checkpointing, this uses signficantly less
    memory than non-checkpointed _gather_reduce.

    JAX/XLA is poor at differentiating iterative algorithms compared to Julia.
    On JAX version 0.7.2, this is slower to differentiate than _gather_reduce.
    On JAX version 0.7.2, enabling CSE did not increase memory.
    """

    def body(i, val):
        c0, c1 = val
        return jnp.take_along_axis(cheb[-i], x_idx, axis=-1) - c1, c0 + c1 * y2

    num_coef = cheb.shape[-1]
    cheb = jnp.moveaxis(cheb, -1, 0)  # to minimize cache misses
    y2 = 2 * y
    c0 = jnp.take_along_axis(cheb[-2], x_idx, axis=-1)
    c1 = jnp.take_along_axis(cheb[-1], x_idx, axis=-1)
    c0, c1 = jax.lax.fori_loop(3, num_coef + 1, body, (c0, c1))
    return c0 + c1 * y


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _intersect2d(o, k, eps):
    """Coordinates yᵢ such that f(x, yᵢ) = k(x).

    Parameters
    ----------
    o : PiecwiseChebyshevSeries
    k : jnp.ndarray
        Shape must broadcast with (..., *cheb.shape[:-1]).
        Specify to find solutions yᵢ to f(x, yᵢ) = k(x). Default 0.
    eps : float
        Absolute tolerance with which to consider value as zero.

    Returns
    -------
    y : jnp.ndarray
        Shape (..., *cheb.shape[:-1], Y - 1).
        Solutions yᵢ of f(x, yᵢ) = k(x), in ascending order,
        are given by ``bijection_from_disc(y,*o.domain)``.
    mask : jnp.ndarray
        Shape y.shape.
        Boolean array into ``y`` indicating whether element is an intersect.
    df_dy : jnp.ndarray
        Shape y.shape.
        ∂f/∂y (x, yᵢ).

    """
    # roots yᵢ of f(x, y) = ∑ₙ₌₀ᴺ⁻¹ αₙ(x) Tₙ(y) - k(x)
    y = _chebroots_vec(subtract_first(o.cheb, k))

    # Intersects must satisfy y ∈ [-1, 1].
    # Pick sentinel such that only distinct roots are considered intersects.
    y = filter_distinct(y, sentinel=-2.0, eps=eps)
    mask = (jnp.abs(y.imag) <= eps) & (jnp.abs(y.real) < 1.0)
    # Ensure y ∈ (-1, 1), i.e. where arccos is differentiable.
    y = jnp.where(mask, y.real, 0.0)

    n = jnp.arange(o.Y)
    # ∂f/∂y = ∑ₙ₌₀ᴺ⁻¹ aₙ(x) n Uₙ₋₁(y)
    df_dy = jnp.einsum(
        "...yn, ...n",
        jnp.sin(n * jnp.arccos(y)[..., None]) / jnp.sqrt(1 - y**2)[..., None],
        o.cheb * n,
    )
    return y, mask, df_dy


@_intersect2d.defjvp
def _intersect2d_jvp(eps, primals, tangents):
    """Implicit function theorem with regularization.

    Regularization used to smooth the discretized system so that it recognizes
    any non-differentiable sample it has observed actually has zero measure in
    the continuous system.

    JAX also does a custom_jvp for the eigenvalue problem, but the approach
    here also diagonalizes the jvp with respect to k, so it is more effecient
    by a factor of the size of k.

    References
    ----------
    Spectrally accurate, reverse-mode differentiable bounce-averaging
    algorithm and its applications.
    Kaya E. Unalmis et al.
    Supplementary information.

    """
    o, k = primals
    do, dk = tangents

    y, mask, df_dy = _intersect2d(o, k, eps)
    n = jnp.arange(o.Y)
    df_do = jnp.einsum("...yn, ...n", jnp.cos(n * jnp.arccos(y)[..., None]), do.cheb)
    dy = jnp.where(
        mask,
        (jnp.expand_dims(dk, -1) - df_do)
        / jnp.where(
            jnp.abs(df_dy) > eps,
            df_dy,
            df_dy + jnp.copysign(eps, df_dy.real),
        ),
        0.0,
    )
    return (y, mask, df_dy), (
        dy,
        jnp.zeros_like(mask, dtype=jax.dtypes.float0),
        # we will always call df_dy = sign(df_dy) on output,
        # so we can skip the computation of this hessian.
        jnp.zeros_like(df_dy),
    )


class PiecewiseChebyshevSeries(Module):
    """Chebyshev series.

    { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
    and Tₙ are Chebyshev polynomials on [−yₘᵢₙ, yₘₐₓ]

    Parameters
    ----------
    cheb : jnp.ndarray
        Shape (..., X, Y).
        Chebyshev coefficients aₙ(x) for f(x, y) = ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y).
    domain : tuple[float]
        Domain for y coordinates. Default is [-1, 1].

    """

    cheb: jax.Array
    domain: tuple[float]

    def __init__(self, cheb, domain=(-1, 1)):
        """Make piecewise series from given Chebyshev coefficients."""
        errorif(domain[0] > domain[-1], msg="Got inverted domain.")
        self.cheb = jnp.atleast_2d(cheb)
        self.domain = domain

    @property
    def X(self):
        """Number of cuts."""
        return self.cheb.shape[-2]

    @property
    def Y(self):
        """Chebyshev spectral resolution."""
        return self.cheb.shape[-1]

    @staticmethod
    def stitch(cheb):
        """Enforce continuity of the piecewise series.

        In some applications, the given piecewise series
        may not be continuous to machine precision due to incomplete convergence
        of some input.
        In that case, this method is useful to enforce continuity exactly
        in the discrete system by adjusting the series in the next partition
        to start where the previous series ends.

        """
        cheb = jnp.atleast_2d(cheb)
        # evaluate at left boundary
        f_0 = cheb[..., ::2].sum(-1) - cheb[..., 1::2].sum(-1)
        # evaluate at right boundary
        f_1 = cheb.sum(-1)
        dfx = f_1[..., :-1] - f_0[..., 1:]  # Δf = f(xᵢ, y₁) - f(xᵢ₊₁, y₀)
        cheb = cheb.at[..., 1:, 0].add(dfx.cumsum(-1))
        return cheb

    def evaluate(self, Y):
        """Evaluate Chebyshev series at Y Chebyshev points.

        Evaluate each function in this set
        { fₓ | fₓ : y ↦ ∑ₙ₌₀ᴺ⁻¹ aₙ(x) Tₙ(y) }
        at y points given by the Y Chebyshev points.

        Parameters
        ----------
        Y : int
            Grid resolution in y direction. Preferably power of 2.

        Returns
        -------
        fq : jnp.ndarray
            Shape (..., X, Y)
            Chebyshev series evaluated at Y Chebyshev points.

        """
        warnif(
            Y < self.Y,
            msg="Frequency spectrum of DCT interpolation will be truncated because "
            "the grid resolution is less than the Chebyshev resolution.\n"
            f"Got Y = {Y} < {self.Y} = self.Y.",
        )
        return idct(dct_from_cheb(self.cheb), type=2, n=Y, axis=-1) * Y

    def _isomorphism_to_C1(self, y):
        """Return coordinates z ∈ ℂ isomorphic to (x, y) ∈ ℂ².

        Maps row x of y to z = y + f(x) where f(x) = x * |domain|.

        Parameters
        ----------
        y : jnp.ndarray
            Shape (..., y.shape[-2], y.shape[-1]).
            Second to last axis iterates the rows.

        Returns
        -------
        z : jnp.ndarray
            Shape y.shape.
            Isomorphic coordinates.

        """
        assert y.ndim >= 2
        z_shift = jnp.arange(y.shape[-2]) * (self.domain[-1] - self.domain[0])
        return y + z_shift[:, None]

    def _isomorphism_to_C2(self, z):
        """Return coordinates (x, y) ∈ ℂ² isomorphic to z ∈ ℂ.

        Returns index x and minimum value y such that
        z = f(x) + y where f(x) = x * |domain|.

        Parameters
        ----------
        z : jnp.ndarray
            Shape z.shape.

        Returns
        -------
        x_idx, y : tuple[jnp.ndarray]
            Shape z.shape.
            Isomorphic coordinates.

        """
        x_idx, y = jnp.divmod(z - self.domain[0], self.domain[-1] - self.domain[0])
        x_idx = x_idx.astype(int)
        y += self.domain[0]
        return x_idx, y

    def eval1d(self, z, cheb=None, loop=False):
        """Evaluate piecewise Chebyshev series at coordinates z.

        Parameters
        ----------
        z : jnp.ndarray
            Shape should broadcast with (*cheb.shape[:-2], z.shape[-1])
            and also have the same dimension as (*cheb.shape[:-2], z.shape[-1]).
            Coordinates in [self.domain[0], ∞).
            The coordinates z ∈ ℝ are assumed isomorphic to (x, y) ∈ ℝ² where
            ``z//domain`` yields the index into the proper Chebyshev series
            along the second to last axis of ``cheb`` and ``z%domain`` is
            the coordinate value on the domain of that Chebyshev series.
        cheb : jnp.ndarray
            Shape (..., X, Y).
            Chebyshev coefficients to use. If not given, uses ``self.cheb``.
        loop : bool
            If ``True``, then uses Clenshaw recursion which is memory efficient.
            If ``False``, then gathers a large block of memory and computes
            a product sum reduction while checkpointing the derivative
            to reduce memory consumption of the Jacobian.

        Returns
        -------
        f : jnp.ndarray
            Chebyshev series evaluated at z.

        """
        if cheb is None:
            cheb = self.cheb
        x_idx, y = self._isomorphism_to_C2(z)
        y = bijection_to_disc(y, *self.domain)

        if loop and self.Y >= 3:
            return _loop(y, cheb, x_idx)

        y = jnp.arccos(y)
        return _gather_reduce(y, cheb, x_idx)

    def intersect1d(self, k=0.0, eps=None, num_intersect=-1):
        """Coordinates z(x, yᵢ) such that fₓ(yᵢ) = k for every x.

        Notes
        -----
        It is recommended to pip install ``orthax`` to use this method.
        If ``orthax`` is not installed, numpy will be used, which is
        less performant.

        Parameters
        ----------
        k : jnp.ndarray
            Shape must broadcast with (..., *cheb.shape[:-2]).
            Specify to find solutions yᵢ to fₓ(yᵢ) = k. Default 0.
        eps : float
            Absolute tolerance with which to consider value as zero.
            Default is near machine epsilon.
        num_intersect : int or None
            Specify to return the first ``num_intersect`` intersects.
            This is useful if ``num_intersect`` tightly bounds the actual number.

            If not specified, then all intersects are returned. If there were fewer
            intersects detected than the size of the last axis of the returned arrays,
            then that axis is padded with zero.

        Returns
        -------
        z1, z2 : tuple[jnp.ndarray]
            Shape broadcasts with (..., *self.cheb.shape[:-2], num intersect).
            Tuple of length two (z1, z2) of coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.

        """
        errorif(
            self.Y < 2,
            NotImplementedError,
            "This method requires a Chebyshev spectral resolution of Y > 1, "
            f"but got Y = {self.Y}.",
        )
        if eps is None:
            eps = max(jnp.finfo(jnp.array(1.0).dtype).eps, 2.5e-12)

        # Add axis to use same k over all Chebyshev series of the piecewise series.
        y, mask, df_dy = _intersect2d(self, jnp.atleast_1d(k)[..., None], eps=eps)
        df_dy = jnp.sign(df_dy)
        y = bijection_from_disc(y, *self.domain)

        # Flatten so that last axis enumerates intersects along the piecewise series.
        y = flatten_mat(self._isomorphism_to_C1(y))
        mask = flatten_mat(mask)
        df_dy = flatten_mat(df_dy)

        z1 = (df_dy <= 0) & mask
        z2 = (df_dy >= 0) & epigraph_and(mask, df_dy)

        sentinel = self.domain[0] - 1.0
        z1 = take_mask(y, z1, size=num_intersect, fill_value=sentinel)
        z2 = take_mask(y, z2, size=num_intersect, fill_value=sentinel)

        mask = (z1 > sentinel) & (z2 > sentinel)
        # Set to zero so integration is over set of measure zero
        # and basis functions are faster to evaluate in downstream routines.
        z1 = jnp.where(mask, z1, 0.0)
        z2 = jnp.where(mask, z2, 0.0)
        return z1, z2

    def _check_shape(self, z1, z2, k):
        """Return shapes that broadcast with (k.shape[0], *self.cheb.shape[:-2], W)."""
        assert z1.shape == z2.shape
        # Ensure batch dim exists and add back dim to broadcast with wells.
        k = atleast_nd(self.cheb.ndim - 1, k)[..., None]
        # Same but back dim already exists.
        z1 = atleast_nd(self.cheb.ndim, z1)
        z2 = atleast_nd(self.cheb.ndim, z2)
        # Cheb has shape    (..., X, Y) and others
        #     have shape (K, ..., W)
        assert z1.ndim == z2.ndim == k.ndim == self.cheb.ndim
        return z1, z2, k

    def check_intersect1d(self, z1, z2, k, *, plot=True, **kwargs):
        """Check that intersects are computed correctly.

        Parameters
        ----------
        z1, z2 : jnp.ndarray
            Shape must broadcast with (*self.cheb.shape[:-2], W).
            Coordinates of intersects.
            The points are ordered and grouped such that the straight line path
            between ``z1`` and ``z2`` resides in the epigraph of f.
        k : jnp.ndarray
            Shape must broadcast with self.cheb.shape[:-2].
            k such that fₓ(yᵢ) = k.
        plot : bool
            Whether to plot the piecewise spline and intersects for the given ``k``.
            For the plotting labels of ρ(l), α(m), it is assumed that the axis that
            enumerates the index l preceds the axis that enumerates the index m.
        kwargs : dict
            Keyword arguments into ``self.plot``.

        Returns
        -------
        plots : list
            Matplotlib (fig, ax) tuples for the 1D plot of each field line.

        """
        kwargs.setdefault("title", r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$")
        title = kwargs.pop("title")
        plots = []

        z1, z2, k = self._check_shape(z1, z2, k)
        mask = (z1 - z2) != 0.0
        z1 = jnp.where(mask, z1, jnp.nan)
        z2 = jnp.where(mask, z2, jnp.nan)

        err_1 = jnp.any(z1 > z2, axis=-1)
        err_2 = jnp.any(z1[..., 1:] < z2[..., :-1], axis=-1)
        f_midpoint = self.eval1d((z1 + z2) / 2, self.cheb[None])
        eps = kwargs.pop("eps", jnp.finfo(jnp.array(1.0).dtype).eps * 10)
        err_3 = jnp.any(f_midpoint > k + eps, axis=-1)
        if not (plot or jnp.any(err_1 | err_2 | err_3)):
            return plots

        cheb = atleast_nd(3, self.cheb)
        mask, z1, z2, f_midpoint = map(atleast_3d_mid, (mask, z1, z2, f_midpoint))
        err_1, err_2, err_3 = map(atleast_2d_end, (err_1, err_2, err_3))

        for l in np.ndindex(cheb.shape[:-2]):
            for p in range(k.shape[0]):
                idx = (p, *l)
                if not (err_1[idx] or err_2[idx] or err_3[idx]):
                    continue
                _z1 = z1[idx][mask[idx]]
                _z2 = z2[idx][mask[idx]]
                if plot:
                    self.plot1d(
                        cheb=cheb[l],
                        z1=_z1,
                        z2=_z2,
                        k=k[idx],
                        title=title
                        + rf" on field line $\rho(l)$, $\alpha(m)$, $(l,m)=${l}",
                        **kwargs,
                    )
                print("      z1    |    z2")
                print(jnp.column_stack([_z1, _z2]))
                assert not err_1[idx], "Intersects have an inversion.\n"
                assert not err_2[idx], "Detected discontinuity.\n"
                assert not err_3[idx], (
                    f"Detected f = {f_midpoint[idx][mask[idx]]} > {k[idx] + eps} = k"
                    "in well, implying the straight line path between z1 and z2 is in"
                    "hypograph(f). Increase spectral resolution.\n"
                )
            idx = (slice(None), *l)
            if plot:
                plots.append(
                    self.plot1d(
                        cheb=cheb[l],
                        z1=z1[idx],
                        z2=z2[idx],
                        k=k[idx],
                        title=title,
                        **kwargs,
                    )
                )
        return plots

    def plot1d(
        self,
        cheb,
        num=5000,
        z1=None,
        z2=None,
        k=None,
        *,
        k_transparency=0.5,
        klabel=r"$k$",
        title=r"Intersects $z$ in epigraph$(f)$ s.t. $f(z) = k$",
        hlabel=r"$z$",
        vlabel=r"$f$",
        show=True,
        include_legend=True,
        return_legend=False,
        legend_kwargs=None,
        **kwargs,
    ):
        """Plot the piecewise Chebyshev series ``cheb``.

        Parameters
        ----------
        cheb : jnp.ndarray
            Shape (X, Y).
            Piecewise Chebyshev series f.
        num : int
            Number of points to evaluate ``cheb`` for plot.
        z1 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y <= 0.
        z2 : jnp.ndarray
            Shape (k.shape[0], W).
            Optional, intersects with ∂f/∂y >= 0.
        k : jnp.ndarray
            Shape (k.shape[0], ).
            Optional, k such that fₓ(yᵢ) = k.
        k_transparency : float
            Transparency of horizontal pitch lines.
        klabel : float
            Label of intersect lines.
        title : str
            Plot title.
        hlabel : str
            Horizontal axis label.
        vlabel : str
            Vertical axis label.
        show : bool
            Whether to show the plot. Default is true.
        include_legend : bool
            Whether to plot the legend. Default is true.

        Returns
        -------
        fig, ax
            Matplotlib (fig, ax) tuple.

        """
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", None))

        legend = {}
        z = jnp.linspace(
            start=self.domain[0],
            stop=self.domain[0] + (self.domain[-1] - self.domain[0]) * self.X,
            num=num,
        )
        _add2legend(legend, ax.plot(z, self.eval1d(z, cheb), label=vlabel, **kwargs))
        _plot_intersect(
            ax=ax,
            legend=legend,
            z1=z1,
            z2=z2,
            k=k,
            k_transparency=k_transparency,
            klabel=klabel,
            hlabel=hlabel,
            **kwargs,
        )
        ax.set_xlabel(hlabel)
        ax.set_ylabel(vlabel)
        ax.set_title(title)

        if include_legend:
            if legend_kwargs is None:
                legend_kwargs = dict(loc="lower right")
            ax.legend(legend.values(), legend.keys(), **legend_kwargs)

        if show:
            plt.show()
            plt.close()
        return (fig, ax, legend) if return_legend else (fig, ax)


def _plot_intersect(
    ax,
    legend,
    z1,
    z2,
    k,
    k_transparency,
    klabel,
    hlabel,
    markersize=plt.rcParams["lines.markersize"] * 3,
    **kwargs,
):
    """Plot intersects on ``ax``."""
    if k is None:
        return

    k = jnp.atleast_1d(jnp.squeeze(k))
    assert k.ndim == 1
    z1, z2 = jnp.atleast_2d(z1, z2)
    assert z1.ndim == z2.ndim >= 2
    assert k.shape[0] == z1.shape[0] == z2.shape[0]
    for p in k:
        _add2legend(
            legend,
            ax.axhline(
                p,
                color="tab:purple",
                alpha=k_transparency,
                label=klabel,
                linestyle="--",
            ),
        )
    for i in range(k.size):
        _z1, _z2 = z1[i], z2[i]
        if _z1.size == _z2.size:
            mask = (_z1 - _z2) != 0.0
            _z1 = _z1[mask]
            _z2 = _z2[mask]
        _add2legend(
            legend,
            ax.scatter(
                _z1,
                jnp.full_like(_z1, k[i]),
                marker="v",
                color="tab:red",
                label=hlabel + r"$_1(w)$",
                s=markersize,
            ),
        )
        _add2legend(
            legend,
            ax.scatter(
                _z2,
                jnp.full_like(_z2, k[i]),
                marker="^",
                color="tab:green",
                label=hlabel + r"$_2(w)$",
                s=markersize,
            ),
        )


def _add2legend(legend, lines):
    """Add lines to legend if it's not already in it."""
    for line in setdefault(lines, [lines], hasattr(lines, "__iter__")):
        label = line.get_label()
        if label not in legend:
            legend[label] = line
