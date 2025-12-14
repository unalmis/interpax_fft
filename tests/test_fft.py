"""Tests for uniform FFT interpolation."""

import numpy as np
import pytest
from jax import config

from interpax_fft import rfft_interp1d, rfft_interp2d

config.update("jax_enable_x64", True)


@pytest.mark.unit
def test_fft_interp1d():
    """Test for 1d Fourier interpolation."""

    def fun(x):
        return 2 * np.sin(1 * x) + 4 * np.cos(3 * x) + 1

    x = {"o": {}, "e": {}}
    x["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    x["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    x["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)
    f1 = {}
    for p in ["o", "e"]:
        f1[p] = {}
        for i in [1, 2]:
            f1[p][i] = fun(x[p][i])

    sx = 0.2
    for sp in ["o", "e"]:  # source parity
        fi = f1[sp][1]
        fs = fun(x[sp][1] + sx)
        np.testing.assert_allclose(
            rfft_interp1d(fi, *fi.shape, sx, dx=x[sp][1][1] - x[sp][1][0]).squeeze(), fs
        )
        for ep in ["o", "e"]:  # eval parity
            for s in ["up", "down"]:  # up or downsample
                if s == "up":
                    xs = 1
                    xe = 2
                else:
                    xs = 2
                    xe = 1
                true = fun(x[ep][xe] + sx)
                interp = rfft_interp1d(
                    f1[sp][xs], x[ep][xe].size, sx, dx=x[sp][xs][1] - x[sp][xs][0]
                ).squeeze()
                np.testing.assert_allclose(interp, true, atol=1e-12, rtol=1e-12)


@pytest.mark.unit
def test_fft_interp2d():
    """Test for 2d Fourier interpolation."""

    def fun2(x, y):
        return (
            2 * np.sin(1 * x[:, None])
            - 1.2 * np.cos(2 * x[:, None])
            + 3 * np.cos(3 * y[None])
            - 2 * np.cos(5 * y[None])
            + 1
        )

    x = {"o": {}, "e": {}}
    y = {"o": {}, "e": {}}
    x["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    x["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    x["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    x["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)
    y["o"][1] = np.linspace(0, 2 * np.pi, 33, endpoint=False)
    y["e"][1] = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    y["o"][2] = np.linspace(0, 2 * np.pi, 133, endpoint=False)
    y["e"][2] = np.linspace(0, 2 * np.pi, 132, endpoint=False)

    f2 = {}
    for xp in ["o", "e"]:
        f2[xp] = {}
        for yp in ["o", "e"]:
            f2[xp][yp] = {}
            for i in [1, 2]:
                f2[xp][yp][i] = {}
                for j in [1, 2]:
                    f2[xp][yp][i][j] = fun2(x[xp][i], y[yp][j])

    shiftx = 0.2
    shifty = 0.3
    for spx in ["o", "e"]:  # source parity x
        for spy in ["o", "e"]:  # source parity y
            fi = f2[spx][spy][1][1]
            fs = fun2(x[spx][1] + shiftx, y[spy][1] + shifty)
            np.testing.assert_allclose(
                rfft_interp2d(
                    fi,
                    *fi.shape,
                    shiftx,
                    shifty,
                    dx=np.diff(x[spx][1])[0],
                    dy=np.diff(y[spy][1])[0]
                ).squeeze(),
                fs,
            )
            for epx in ["o", "e"]:  # eval parity x
                for epy in ["o", "e"]:  # eval parity y
                    for sx in ["up", "down"]:  # up or downsample x
                        if sx == "up":
                            xs = 1
                            xe = 2
                        else:
                            xs = 2
                            xe = 1
                        for sy in ["up", "down"]:  # up or downsample y
                            if sy == "up":
                                ys = 1
                                ye = 2
                            else:
                                ys = 2
                                ye = 1

                            true = fun2(x[epx][xe] + shiftx, y[epy][ye] + shifty)
                            interp = rfft_interp2d(
                                f2[spx][spy][xs][ys],
                                x[epx][xe].size,
                                y[epy][ye].size,
                                shiftx,
                                shifty,
                                dx=x[spx][xs][1] - x[spx][xs][0],
                                dy=y[spy][ys][1] - y[spy][ys][0],
                            ).squeeze()
                            np.testing.assert_allclose(
                                interp, true, atol=1e-12, rtol=1e-12
                            )
