
############
interpax_fft
############
|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

interpax_fft is a library for Fourier interpolation and function approximation using JAX.

Basic methods for Fourier interpolation.


Installation
============

interpax_fft is installable with `pip`:

.. code-block:: console

    pip install interpax_fft



Usage
=====

.. code-block:: python

    import jax.numpy as jnp
    import numpy as np
    from interpax import interp1d

    xp = jnp.linspace(0, 2 * np.pi, 100)
    xq = jnp.linspace(0, 2 * np.pi, 10000)
    f = lambda x: jnp.sin(x)
    fp = f(xp)

    fq = interp1d(xq, xp, fp, method="cubic")
    np.testing.assert_allclose(fq, f(xq), rtol=1e-6, atol=1e-5)


For full details of various options see the `API documentation <https://interpax_fft.readthedocs.io/en/latest/api.html>`__


.. |License| image:: https://img.shields.io/github/license/unalmis/interpax_fft?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/unalmis/interpax_fft/blob/master/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/706703896.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10028967
    :alt: DOI

.. |Docs| image:: https://img.shields.io/readthedocs/interpax?logo=Read-the-Docs
    :target: https://interpax.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |UnitTests| image:: https://github.com/unalmis/interpax_fft/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/unalmis/interpax_fft/actions/workflows/unittest.yml
    :alt: UnitTests

.. |Codecov| image:: https://codecov.io/github/unalmis/interpax_fft/graph/badge.svg?token=MB11I7WE3I
    :target: https://codecov.io/github/unalmis/interpax_fft
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/unalmis/interpax
    :target: https://github.com/unalmis/interpax_fft/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/interpax_fft
    :target: https://pypi.org/project/interpax_fft/
    :alt: Pypi
