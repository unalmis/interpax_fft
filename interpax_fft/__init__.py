"""interpax_fft: Fourier interpolation and function approximation with JAX."""

from . import _version
from ._fft import irfft_interp1d, irfft_interp2d, rfft_interp1d, rfft_interp2d
from ._interp_utils import (
    cheb_from_dct,
    cheb_pts,
    dct_from_cheb,
    fourier_pts,
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

__all__ = [
    "cheb_pts",
    "fourier_pts",
    "interp_rfft",
    "irfft_mmt",
    "irfft_mmt_pos",
    "ifft_mmt",
    "interp_rfft2",
    "irfft2_mmt_pos",
    "rfft2_vander",
    "rfft2_modes",
    "cheb_from_dct",
    "dct_from_cheb",
    "interp_dct",
    "idct_mmt",
    "rfft_to_trig",
    "trig_vander",
    "rfft_interp1d",
    "rfft_interp2d",
    "irfft_interp1d",
    "irfft_interp2d",
]

__version__ = _version.get_versions()["version"]
