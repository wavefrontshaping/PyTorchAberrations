"""Invariants of the aberration layers.

Each test here corresponds to a bug that reached the repository, so they are cheap
regression guards rather than exhaustive coverage:

* the Zernike table (j = 7, 8 were built at the wrong azimuthal order);
* the identity at initialisation (affine_grid and grid_sample disagreed on
  align_corners, so an untrained layer resampled its input);
* ComplexDeformation running at all (view_as_complex needs a contiguous input).
"""

import numpy as np
import pytest
import torch

from PyTorchAberrations.aberration_layers import (ComplexDeformation, ComplexScaling,
                                                  ComplexZernike, zernike_Z)
from PyTorchAberrations.aberration_models import Aberration, AberrationModes


N_MODES, N_IN, N_OUT = 6, 35, 41


def _coords(n):
    ax = np.arange(0, 2, 2. / n) - (1. + 1. / n)
    X, Y = np.meshgrid(ax, ax, indexing='ij')
    return X, Y


def _standard_zernike(n):
    """The OSA/ANSI polynomials in the slots zernike_Z claims to fill."""
    X, Y = _coords(n)
    R, T = np.hypot(X, Y), np.arctan2(Y, X)
    return {
        1: 2 * X,
        2: 2 * Y,
        3: 6**.5 * R**2 * np.sin(2 * T),
        4: 3**.5 * (2 * R**2 - 1),
        5: 6**.5 * R**2 * np.cos(2 * T),
        6: 8**.5 * R**3 * np.sin(3 * T),
        7: 8**.5 * (3 * R**3 - 2 * R) * np.sin(T),      # vertical coma, m = -1
        8: 8**.5 * (3 * R**3 - 2 * R) * np.cos(T),      # horizontal coma, m = +1
        9: 8**.5 * R**3 * np.cos(3 * T),
        10: 10**.5 * R**4 * np.sin(4 * T),
        11: 10**.5 * (4 * R**4 - 3 * R**2) * np.sin(2 * T),
        12: 5**.5 * (6 * R**4 - 6 * R**2 + 1),
        13: 10**.5 * (4 * R**4 - 3 * R**2) * np.cos(2 * T),
        14: 10**.5 * R**4 * np.cos(4 * T),
    }


def _random_stack(k, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(k, n, n, generator=g)
            + 1j * torch.randn(k, n, n, generator=g)).to(torch.complex128)


@pytest.mark.parametrize('j', range(1, 15))
def test_zernike_matches_standard_definition(j):
    """Every polynomial must be the standard Z_n^m, not merely something smooth."""
    n = 201
    X, Y = _coords(n)
    disk = np.hypot(X, Y) <= 1
    got = zernike_Z(j, torch.from_numpy(X), torch.from_numpy(Y)).numpy()[disk]
    want = _standard_zernike(n)[j][disk]
    corr = abs(np.dot(got, want)) / np.linalg.norm(got) / np.linalg.norm(want)
    assert corr > 0.999, f'Z_{j} does not match the standard definition (corr {corr:.4f})'


def test_zernike_set_is_orthogonal_on_the_disk():
    """Two terms at the same azimuthal order would show up as a large overlap."""
    n = 201
    X, Y = _coords(n)
    disk = np.hypot(X, Y) <= 1
    V = np.stack([zernike_Z(j, torch.from_numpy(X), torch.from_numpy(Y)).numpy()[disk]
                  for j in range(1, 15)])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    G = V @ V.T
    off = np.abs(G - np.diag(np.diag(G)))
    assert off.max() < 0.05, f'Zernike terms overlap (max {off.max():.3f})'


@pytest.mark.parametrize('layer', [ComplexScaling, ComplexDeformation, ComplexZernike])
def test_layer_is_identity_at_initialisation(layer):
    """Every layer initialises at a neutral value, so it must return its input."""
    mod = layer(j=4) if layer is ComplexZernike else layer()
    x = _random_stack(3, 32)
    out = mod(x)
    assert out.shape == x.shape
    err = (torch.norm(out - x) / torch.norm(x)).item()
    assert err < 1e-6, f'{layer.__name__} is not the identity at init (rel err {err:.2e})'


@pytest.mark.parametrize('deformation', ['none', 'scaling', 'single'])
@pytest.mark.parametrize('padding_coeff', [0.0, 0.05, 0.5])
def test_untrained_model_is_identity(deformation, padding_coeff):
    """An untrained Aberration must be a no-op: pad, transform, crop returns the input."""
    model = Aberration(N_IN,
                       list_zernike_ft=list(range(9)),
                       list_zernike_direct=list(range(14)),
                       padding_coeff=padding_coeff,
                       deformation=deformation)
    x = _random_stack(N_MODES, N_IN)
    err = (torch.norm(model(x) - x) / torch.norm(x)).item()
    assert err < 1e-6, f'untrained model is not the identity (rel err {err:.2e})'


def test_aberration_modes_returns_output_then_input():
    model = AberrationModes(inpoints=N_IN, onpoints=N_OUT, padding_coeff=0.05,
                            list_zernike_ft=list(range(9)),
                            list_zernike_direct=list(range(14)))
    xi, xo = _random_stack(N_MODES, N_IN), _random_stack(N_MODES, N_OUT, seed=1)
    out, inp = model(xi, xo)
    assert out.shape == xo.shape and inp.shape == xi.shape
    assert torch.norm(inp - xi) / torch.norm(xi) < 1e-6
    assert torch.norm(out - xo) / torch.norm(xo) < 1e-6


def test_zernike_parameter_has_an_effect_and_a_gradient():
    mod = ComplexZernike(j=4)
    x = _random_stack(3, 32)
    with torch.no_grad():
        mod.alpha.fill_(0.5)
    out = mod(x)
    assert (torch.norm(out - x) / torch.norm(x)).item() > 1e-3

    mod.alpha.grad = None
    ref = _random_stack(3, 32, seed=2)
    ((mod(x) * ref.conj()).sum().abs()**2).backward()
    assert mod.alpha.grad is not None and mod.alpha.grad.abs().item() > 0


def test_scaling_parameter_changes_the_field():
    mod = ComplexScaling()
    x = _random_stack(3, 32)
    with torch.no_grad():
        mod.theta.fill_(0.2)
    assert (torch.norm(mod(x) - x) / torch.norm(x)).item() > 1e-3


@pytest.mark.xfail(reason='ComplexDeformation masks theta by [1,0,0,0,1,0], so the two '
                          'shears and the two shifts have exactly zero gradient and '
                          'cannot be learned, despite the docstring claiming 6 '
                          'parameters. Unmasking changes the capacity of the model, so '
                          'it is left as a deliberate decision.',
                   strict=True)
@pytest.mark.parametrize('index', [1, 2, 3, 5])
def test_deformation_shear_and_shift_are_live(index):
    mod = ComplexDeformation()
    x = _random_stack(3, 32)
    with torch.no_grad():
        mod.theta.zero_()
        mod.theta[index] = 0.3
    assert (torch.norm(mod(x) - x) / torch.norm(x)).item() > 1e-6
