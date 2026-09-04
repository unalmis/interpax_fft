=================
API Documentation
=================

Interpolation with fast Fourier transform (FFT)
-----------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax_fft.rfft_interp1d
    interpax_fft.rfft_interp2d
    interpax_fft.irfft_interp1d
    interpax_fft.irfft_interp2d

Interpolation with matrix multiplication transform (MMT)
--------------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax_fft.idct_mmt
    interpax_fft.ifft_mmt
    interpax_fft.interp_dct
    interpax_fft.interp_rfft
    interpax_fft.interp_rfft2
    interpax_fft.irfft2_mmt_pos
    interpax_fft.irfft_mmt
    interpax_fft.irfft_mmt_pos
    interpax_fft.rfft2_modes
    interpax_fft.rfft2_vander
    interpax_fft.rfft_to_trig
    interpax_fft.trig_vander

Function approximation with series expansions
---------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax_fft.DoubleChebyshevSeries
    interpax_fft.FourierChebyshevSeries
    interpax_fft.PiecewiseChebyshevSeries

Useful utilities for interpolation, root finding, etc.
------------------------------------------------------

.. autosummary::
    :toctree: _api/
    :recursive:

    interpax_fft.cheb_from_dct
    interpax_fft.cheb_pts
    interpax_fft.dct_from_cheb
    interpax_fft.epigraph_and
    interpax_fft.fourier_pts
    interpax_fft.take_mask
