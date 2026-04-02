"""interpax_fft: Fourier interpolation and function approximation with JAX."""

from . import _version
from ._fft import irfft_interp1d, irfft_interp2d, rfft_interp1d, rfft_interp2d
from ._mmt import (
    idct_mmt,
    ifft_mmt,
    interp_dct,
    interp_rfft,
    interp_rfft2,
    irfft2_mmt_pos,
    irfft_mmt,
    irfft_mmt_pos,
    rfft2_modes,
    rfft2_vander,
    rfft_to_trig,
    trig_vander,
)
from ._series import (
    DoubleChebyshevSeries,
    FourierChebyshevSeries,
    PiecewiseChebyshevSeries,
)
from ._utils_public import (
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    epigraph_and,
    fourier_pts,
    take_mask,
)

__all__ = [
    "DoubleChebyshevSeries",
    "FourierChebyshevSeries",
    "PiecewiseChebyshevSeries",
    "cheb_from_dct",
    "cheb_pts",
    "dct_from_cheb",
    "epigraph_and",
    "fourier_pts",
    "idct_mmt",
    "ifft_mmt",
    "interp_dct",
    "interp_rfft",
    "interp_rfft2",
    "irfft2_mmt_pos",
    "irfft_interp1d",
    "irfft_interp2d",
    "irfft_mmt",
    "irfft_mmt_pos",
    "rfft2_modes",
    "rfft2_vander",
    "rfft_interp1d",
    "rfft_interp2d",
    "rfft_to_trig",
    "take_mask",
    "trig_vander",
]

__version__ = _version.get_versions()["version"]
