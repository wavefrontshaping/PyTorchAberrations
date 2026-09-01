"""Regression tests for `PyTorchAberrations.mode_masks`.

Each guards a property the module is only useful if it has, and each of them was wrong
at some point in the writing.

No dataset is needed: a synthetic band-limited "mode set" on a coarse grid exercises the
same geometry as the real one.
"""

import numpy as np
import pytest
import torch

from PyTorchAberrations.aberration_models import AberrationModes
from PyTorchAberrations.cost_functions import normalize
from PyTorchAberrations.mode_masks import ModeMasks, resample_field, sample_modes
from PyTorchAberrations.scaling_functions import resize_modes

N_HD = 128
N_NATIVE = 35
ZOOM = 0.27
NMODES = 6
PADDING_COEFF = 0.05


@pytest.fixture(scope='module')
def profiles():
    """A band-limited complex mode set on the high-definition grid."""
    rng = np.random.default_rng(0)
    f = np.fft.fftfreq(N_HD)
    R = np.hypot(*np.meshgrid(f, f, indexing='ij'))
    # cutoff chosen so the native grid sees about the same fraction of its Nyquist band
    # as the bundled fiber modes do
    spec = (rng.normal(size=(NMODES, N_HD, N_HD))
            + 1j * rng.normal(size=(NMODES, N_HD, N_HD))) * (R <= 0.06)
    p = np.fft.ifft2(spec, axes=(-2, -1))
    return p / np.linalg.norm(p.reshape(NMODES, -1), axis=1).reshape(-1, 1, 1)


def fitted_model(seed=0):
    """An `AberrationModes` with non-trivial parameters, as after a fit."""
    torch.manual_seed(seed)
    model = AberrationModes(inpoints=N_NATIVE, onpoints=N_NATIVE,
                            padding_coeff=PADDING_COEFF,
                            list_zernike_ft=list(range(5)),
                            list_zernike_direct=list(range(7)),
                            deformation='scaling')
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.05 * torch.randn_like(p))
    return model


def masks(model, profiles, **kw):
    native = {'in': (N_NATIVE, ZOOM), 'out': (N_NATIVE, ZOOM)}
    return ModeMasks(model, profiles, native, padding_coeff=PADDING_COEFF,
                     conj_out=False, **kw)


def rel(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def shape_err(a, b):
    """Difference of shape only.

    Each basis is normalised over the grid it was built on, so the same physical field
    comes back `s` times smaller on an `s` times finer grid. Comparing across
    resolutions therefore means comparing direction, not amplitude.
    """
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return np.linalg.norm(a / np.linalg.norm(a) - b / np.linalg.norm(b))


def native_offset(n, s):
    """Index of the first native sample on an `s` times finer grid."""
    return (s * n) // 2 - s * (n // 2)


def test_native_sampling_matches_resize_modes(profiles):
    """`sample_modes` must lay down the grid the fit actually used.

    `resize_modes` zooms with `scipy.ndimage.zoom`, whose effective pitch is
    ``(n_hd - 1) / (n_zoom - 1)`` rather than ``1 / zoom``, and then crops or pads with a
    floor-divided offset. Sampling on a nominally clean centred grid instead gives a
    *different physical field* -- a 56 % error on the bundled dataset, which looks
    entirely plausible until compared.
    """
    ours = sample_modes(profiles, N_NATIVE, N_NATIVE, ZOOM)
    theirs = resize_modes(profiles, ZOOM, N_NATIVE)
    assert rel(ours, theirs) < 1e-9, rel(ours, theirs)


def test_basis_reproduces_the_model_natively(profiles):
    """At the native resolution the module must return the change-of-basis matrix the
    model itself produces -- that is what makes it a valid extrapolation."""
    model = fitted_model()
    mm = masks(model, profiles)

    native = resize_modes(profiles, ZOOM, N_NATIVE)
    with torch.no_grad():
        out, inp = model(torch.from_numpy(native), torch.from_numpy(native))
    ref_in = normalize(inp).numpy()
    ref_out = normalize(out).numpy()

    assert rel(mm.basis('in'), ref_in) < 1e-5
    assert rel(mm.basis('out'), ref_out) < 1e-5


def test_theoretical_basis_is_resolution_independent(profiles):
    """Without any aberration, refining the grid must return the same physical field:
    the native samples reappear at ``[(s-1)//2::s, (s-1)//2::s]``."""
    mm = masks(fitted_model(), profiles)
    lo = mm.ideal_basis('in')
    for s in (3, 5):
        hi = mm.ideal_basis('in', s * N_NATIVE)
        off = native_offset(N_NATIVE, s)
        sub = hi[:, off::s, off::s]
        sub = sub / np.linalg.norm(sub.reshape(NMODES, -1), axis=1).reshape(-1, 1, 1)
        assert rel(sub, lo) < 1e-9, (s, rel(sub, lo))


def test_zernike_axes_track_the_physical_coordinate():
    """The heart of the resolution transfer, tested as pure geometry.

    A Zernike coefficient means a physical aberration. On a grid `s` times finer over the
    same field of view, the layer's normalised coordinate must therefore land on the same
    value at the same *physical* place. The two planes rescale differently:

    * the **direct** plane keeps its physical extent, so native sample `k` reappears at
      fine sample ``s*k + (s-1)//2``;
    * the **Fourier** plane keeps its frequency *resolution* and widens its band by `s`,
      so native bin `k` reappears one-for-one at ``npad//2 + (k - npad_native//2)`` --
      not `s` bins apart. Rescaling this one as if it were the direct plane is the 69 %
      error the module docstring quotes.
    """
    from PyTorchAberrations.mode_masks import _direct_axis, _ft_axis

    npad_native, s = 37, 3
    npad = s * npad_native

    direct_native = _direct_axis(npad_native, npad_native, 1.0).numpy()
    direct_fine = _direct_axis(npad, npad_native, float(s)).numpy()
    for k in range(npad_native):
        assert direct_fine[s * k + (s - 1) // 2] == pytest.approx(direct_native[k])

    ft_native = _ft_axis(npad_native, npad_native, 1.0).numpy()
    ft_fine = _ft_axis(npad, npad_native, float(s)).numpy()
    for k in range(npad_native):
        j = npad // 2 + (k - npad_native // 2)
        assert ft_fine[j] == pytest.approx(ft_native[k])

    # and the band really is s times wider: the native domain occupies 1/s of it
    assert np.ptp(ft_fine) == pytest.approx(s * np.ptp(ft_native), rel=0.02)


def test_aberration_transfers_across_resolutions(profiles):
    """End to end, with the deformation off so only the Zernike terms are exercised.

    The bound is loose on purpose. The residual depends on how much of the mode set sits
    near the native Nyquist limit -- a direct-plane phase screen broadens the spectrum,
    and what the coarse grid aliases back into its band the fine grid resolves outside
    it. The tight version of this check is in `examples/Demo_mode_masks.ipynb`, on real
    modes, where it comes out at 3e-3.
    """
    model = fitted_model()
    with torch.no_grad():
        for side in (model.abberation_input, model.abberation_output):
            side.deformation.theta.zero_()

    mm = masks(model, profiles)
    lo, s = mm.basis('in'), 3
    hi = mm.basis('in', s * N_NATIVE)
    off = native_offset(N_NATIVE, s)
    assert shape_err(hi[:, off::s, off::s], lo) < 0.1


def test_mask_round_trip(profiles):
    """`modes_from_mask` inverts `mask_from_modes` at the native resolution."""
    mm = masks(fitted_model(), profiles)
    rng = np.random.default_rng(1)
    a = rng.normal(size=NMODES) + 1j * rng.normal(size=NMODES)
    for side in ('in', 'out'):
        field = mm.mask_from_modes(a, side)
        assert field.shape == (N_NATIVE, N_NATIVE)
        back = mm.modes_from_mask(field, side, method='pinv')
        assert rel(back, a) < 1e-8, (side, rel(back, a))


def test_mask_spans_the_same_area(profiles):
    """A mask asked for at a finer resolution is the same field, not a bigger one, and
    not an interpolation of the coarse one."""
    mm = masks(fitted_model(), profiles)
    rng = np.random.default_rng(2)
    a = rng.normal(size=NMODES) + 1j * rng.normal(size=NMODES)
    s = 3
    lo = mm.mask_from_modes(a, 'in')
    hi = mm.mask_from_modes(a, 'in', resolution=s * N_NATIVE)
    assert hi.shape == (s * N_NATIVE,) * 2

    off = native_offset(N_NATIVE, s)
    sub = hi[off::s, off::s]
    # amplitude: each basis is normalised over its own grid, so the finer field is
    # smaller by exactly the sampling density
    assert np.linalg.norm(sub) / np.linalg.norm(lo) == pytest.approx(1 / s, rel=0.05)
    # shape: the residual is the deformation layer's bilinear interpolation on the
    # coarse grid (see the module docstring); a wrong field of view would be far larger
    assert shape_err(sub, lo) < 0.2


def test_resample_field_keeps_the_centre():
    """`resample_field` must anchor on the array centre, not on index 0.

    Plain Fourier zero-padding shifts the grid by (s-1)/2 pixels, which makes two
    identical fields look about 35 % apart -- hence the `ifftshift` before the transform.
    """
    rng = np.random.default_rng(3)
    for n in (16, 17):                     # even and odd: the offset differs
        s = 3
        f = np.fft.fftfreq(n)
        R = np.hypot(*np.meshgrid(f, f, indexing='ij'))
        spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * (R <= 0.15)
        field = np.fft.ifft2(spec)

        up = resample_field(field, s * n)
        off = native_offset(n, s)
        assert rel(up[off::s, off::s], field) < 1e-9, n
        assert rel(resample_field(up, n), field) < 1e-9, n
        # assuming the odd-grid offset on an even grid is a silent 20 % error
        if n % 2 == 0:
            assert rel(up[(s - 1) // 2::s, (s - 1) // 2::s], field) > 0.1


def test_info_reports_the_geometry(profiles):
    mm = masks(fitted_model(), profiles)
    info = mm.info('in', 3 * N_NATIVE)
    assert info['native_resolution'] == N_NATIVE
    assert info['resolution'] == 3 * N_NATIVE
    assert info['oversampling'] == pytest.approx(3.0)
    # the padded array must keep the same physical ratio as the native one
    assert info['padded'] / info['resolution'] == pytest.approx(
        info['padded_native'] / info['native_resolution'], rel=0.02)
