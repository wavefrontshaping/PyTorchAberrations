"""
Modulator-plane masks from mode amplitudes, at any resolution.

Once an :class:`~PyTorchAberrations.aberration_models.AberrationModes` model has been
fitted (or its parameters set by hand), this module turns a vector of mode amplitudes
into the complex field to display in the modulator plane -- on a grid of any size,
spanning the same physical area as the native pixel basis the model was fitted on.

    from PyTorchAberrations.mode_masks import ModeMasks

    mm = ModeMasks.from_tm(model, profiles_hd, TM_pix, pola_inout=(2, 2),
                           padding_coeff=0.05, conj_out=True)

    field = mm.mask_from_modes(coeffs, side='in',  resolution=105)   # (105, 105) complex
    field = mm.mask_from_modes(coeffs, side='out', resolution=205)   # (205, 205) complex
    coeffs_back = mm.modes_from_mask(field, side='out')

`side` selects which of the two `Aberration` sub-models is applied: the input and the
output aberrations are different, and so are the native grids (35x35 and 41x41 for the
bundled data).

See `examples/Demo_mode_masks.ipynb`.

Conventions
-----------
The cost function used to fit the model is

    TM_modes = modes_out @ TM_pix @ modes_in^H

so a vector of mode amplitudes ``a`` corresponds to the pixel-basis field

    u = modes^H a          i.e.   u = sum_k a_k * conj(modes[k])          (synthesis)

and the inverse, for a basis with orthonormal rows, is

    a = modes @ u                                                        (analysis)

``mask_from_modes`` and ``modes_from_mask`` implement exactly these.

Why the resolution change is not just an interpolation
------------------------------------------------------
Two things have to be got right, and both are invisible until measured.

**The sampling grid.** ``scaling_functions.resize_modes`` builds the native basis by
zooming with ``scipy.ndimage.zoom`` and then cropping or padding to size. That leaves an
effective pitch of ``(n_hd - 1) / (n_zoom - 1)`` -- not ``1 / zoom``, a 3.7 % difference
on the bundled data -- and an origin displaced by a floor-divided offset. Sampling the
high-definition profiles on a "clean" centred grid instead gives a *different physical
field*: 56 % away from the basis the model was actually fitted on. :func:`sample_modes`
therefore reproduces that grid exactly at the native resolution and subdivides it.

**The Zernike coordinates.** The layers normalise their coordinates to the array they act
on, so the same coefficient means a different physical aberration on a different grid.
Refining by a factor ``s`` over a fixed field of view

* leaves the **direct**-plane coordinates unchanged in physical terms, but shifts the
  module's grid origin (its grid is centred one pixel off the array centre, and that
  offset is ``1/npad`` in normalised units, so it changes with the grid);
* widens the **Fourier**-plane Nyquist band by ``s``, so the pupil occupies ``1/s`` of
  the normalised domain.

Both are handled by rebuilding each axis from the physical mapping (`_direct_axis` /
`_ft_axis`), which reduce to the module's own grid at ``s = 1``.

What the residuals are
----------------------
Measured on the bundled dataset with a model fitted by
`examples/Demo_correction_aberration.ipynb`, comparing bases on the aligned native
sub-grid (``field[(s-1)//2::s, (s-1)//2::s]`` for odd `s`):

=========================================================  ========
theoretical basis, s = 3 against s = 1                      1.3e-16
**fitted basis, s = 1 against the model's own output**      2.8e-07
fitted basis, s = 3 against s = 1                           3.0e-02
  ... of which the direct-plane Zernike terms contribute    2.4e-15
  ... the Fourier-plane Zernike terms                       3.4e-03
  ... the deformation layer                                 3.0e-02
fitted basis, s = 5 against s = 3                           1.5e-02
=========================================================  ========

The second line is the one to rely on: at the native resolution this module reproduces
the change-of-basis matrix the fit produced, to seven digits.

The 3 % at ``s = 3`` is **the native model's error, not this module's**. It comes from
the deformation layer: ``grid_sample`` has to guess the field between samples, and a
37x37 padded array gives it much less to work with than a 111x111 one. The evidence is
the last line -- the refined grids agree with each other better than any of them agrees
with ``s = 1``, so they converge on a common answer and the coarse one is the outlier.
Before 0.3 this figure was 5 %, when the deformation interpolated bilinearly.
"""

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from PyTorchAberrations.aberration_layers import zernike_Z


__all__ = ['ModeMasks', 'sample_modes', 'rms_width', 'resample_field']


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def resample_field(field, resolution):
    """Band-limited resample of a field between two grids sharing the same centre.

    Use this to compare masks computed at different resolutions, or to bring a
    high-resolution mask back down. Note the `ifftshift` before the forward transform:
    plain Fourier zero-padding anchors the samples at index 0 instead of the array
    centre, which offsets the grid by (s-1)/2 pixels and makes two otherwise identical
    fields look ~35 % different.

    On a grid `s` times finer the original samples land at ``field[off::s, off::s]`` with
    ``off = (s*n)//2 - s*(n//2)``, matching `sample_modes`. That reduces to the familiar
    ``(s-1)//2`` when `n` is **odd**; for even `n` the centre convention puts it at 0
    instead, which is a silent 20 % error if assumed away.
    """
    field = np.asarray(field)
    n = field.shape[-1]
    m = int(resolution)
    spec = np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(field, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    if m >= n:
        out = np.zeros(field.shape[:-2] + (m, m), dtype=complex)
        off = (m - n) // 2
        out[..., off:off + n, off:off + n] = spec
    else:
        off = (n - m) // 2
        out = spec[..., off:off + m, off:off + m]
    return np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(out, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1)) * (m * m) / (n * n)


def rms_width(img):
    """RMS width of a real image about its centroid, in pixels.

    Same definition as ``scaling_functions.compute_rms_width`` in the
    PyTorchAberrations repository, restated here so this module stands alone.
    """
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape[-2:]
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    norm = np.mean(img)
    cx = np.mean(X * img) / norm
    cy = np.mean(Y * img) / norm
    return float(np.sqrt(np.mean(((X - cx)**2 + (Y - cy)**2) * img) / norm))


def sample_modes(profiles_2d, resolution, native_resolution, native_zoom, order=3):
    """Resample high-definition mode profiles onto a grid of `resolution` points
    spanning the same physical area as the native grid.

    `native_zoom` is the factor that maps the high-definition grid onto the native pixel
    grid (``w_native / w_hd`` in the RMS-width matching of ``get_inout_modes``).

    Unlike ``scaling_functions.resize_modes``, the sampling positions are given
    explicitly, so the modes sit at the same physical position at every resolution.
    ``resize_modes`` builds the grid by zooming and then padding to size with a
    floor-divided offset, which leaves a half-pixel shift that varies with the
    resolution -- enough to decorrelate the bases between two grids.

    Returns
    -------
    array, shape (nmodes, resolution, resolution), each mode normalised to unit L2 norm.
    """
    profiles_2d = np.asarray(profiles_2d)
    nmodes, n_hd, _ = profiles_2d.shape

    # Anchor on the grid ``resize_modes`` lays down at the native resolution, then
    # subdivide *that*. scipy's zoom maps output index j to input j*(n_hd-1)/(n_zoom-1),
    # and the crop-or-pad to `size` shifts the origin by a floor-divided offset -- which
    # is where the half-pixel wander comes from. Reproducing it here is what makes the
    # native call return the very basis the model was fitted on, bit for bit, instead of
    # a field shifted by a fraction of a pixel.
    n_zoom = int(np.round(n_hd * native_zoom))
    pitch_native = (n_hd - 1) / (n_zoom - 1)
    # crop start when n_zoom > native_resolution, -pad_left when it is smaller; floor
    # division gives both
    off = (n_zoom - native_resolution) // 2
    centre = (off + (native_resolution - 1) / 2.0) * pitch_native

    # same physical area. Written as a ratio first: at `resolution == native_resolution`
    # that ratio is exactly 1.0, so the native pitch is reproduced bit for bit. Going
    # through ``native_resolution * pitch_native / resolution`` instead costs one ulp,
    # which is enough to push a sample sitting exactly on the array edge to -7e-15 --
    # outside, where map_coordinates returns 0 and the first row and column come back
    # blank.
    pitch = pitch_native * (native_resolution / resolution)
    axis = centre + (np.arange(resolution) - (resolution - 1) / 2.0) * pitch

    YY, XX = np.meshgrid(axis, axis, indexing='ij')
    coords = np.array([YY.ravel(), XX.ravel()])

    out = np.empty((nmodes, resolution, resolution), dtype=complex)
    for k in range(nmodes):
        re = map_coordinates(profiles_2d[k].real, coords, order=order, mode='constant')
        im = map_coordinates(profiles_2d[k].imag, coords, order=order, mode='constant')
        out[k] = (re + 1j * im).reshape(resolution, resolution)

    norms = np.linalg.norm(out.reshape(nmodes, -1), axis=1).reshape(-1, 1, 1)
    return out / norms


def _padding(resolution, native_resolution, padding_coeff):
    """Padding that keeps the padded array in the same physical ratio as the native one.

    The model uses ``int(padding_coeff * N)``; reproducing that ratio (rather than the
    coefficient) is what keeps the padded field of view fixed as the grid is refined.
    """
    npad_native = native_resolution + 2 * int(padding_coeff * native_resolution)
    ratio = npad_native / native_resolution
    pad = int(round((ratio * resolution - resolution) / 2))
    return pad, resolution + 2 * pad, npad_native


def _direct_axis(npad, npad_native, scale, dtype=torch.float64):
    """The native direct-plane Zernike coordinate, evaluated on the target grid.

    ``ComplexZernike`` builds ``arange(0, 2, 2/n) - (1 + 1/n)``, i.e. the coordinate
    origin sits at index ``(n+1)/2`` while the array centre is at ``(n-1)/2`` -- one
    pixel off. Reproducing that offset *in physical units* is what makes a coefficient
    fitted natively mean the same thing here. Reduces to the module's grid at scale = 1.
    """
    i = torch.arange(npad, dtype=dtype)
    return 2. * ((i - (npad - 1) / 2.) / scale - 1.) / npad_native


def _ft_axis(npad, npad_native, scale, dtype=torch.float64):
    """The native Fourier-plane Zernike coordinate, evaluated on the target grid.

    The normalised domain spans the Nyquist band, which is set by the pixel pitch: a grid
    `scale` times finer covers a band `scale` times wider, so the region the fit
    constrains shrinks to 1/scale of the domain. Reduces to the module's grid at
    scale = 1.
    """
    i = torch.arange(npad, dtype=dtype)
    return 2. * scale * (i - npad // 2) / npad - 2. / npad_native


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class ModeMasks:
    """Mode amplitudes -> modulator-plane field, at any resolution.

    Parameters
    ----------
    model : AberrationModes
        A trained model (or one whose parameters were set by hand). Its
        ``abberation_input`` / ``abberation_output`` sub-models are used for
        ``side='in'`` / ``side='out'``.
    profiles_hd : array (nmodes, n_hd, n_hd) or (nmodes, n_hd**2)
        The mode profiles on the high-definition grid.
    native : dict
        ``{'in': (N_in, zoom_in), 'out': (N_out, zoom_out)}`` -- the native pixel grid
        size and the high-definition-to-native zoom for each side.
    padding_coeff : float
        The value used when the model was built.
    conj_out : bool
        Whether the output basis was conjugated before fitting (the updated demo does
        ``modes_out = modes_out.conj()``). Applied to the output side here too.
    deform_interp : {'bicubic', 'bilinear'}
        Interpolation for the deformation layer. The default matches what
        `ComplexScaling` uses, which is what makes the native reconstruction exact;
        change it only to reproduce a model fitted before 0.3.
    """

    def __init__(self, model, profiles_hd, native, padding_coeff=0.05,
                 conj_out=True, deform_interp='bicubic'):
        profiles_hd = np.asarray(profiles_hd)
        if profiles_hd.ndim == 2:
            n_hd = int(round(np.sqrt(profiles_hd.shape[1])))
            profiles_hd = profiles_hd.reshape(-1, n_hd, n_hd)
        self.profiles = profiles_hd
        self.nmodes = profiles_hd.shape[0]
        self.n_hd = profiles_hd.shape[-1]

        self.aber = {'in': model.abberation_input, 'out': model.abberation_output}
        self.native = dict(native)
        self.padding_coeff = padding_coeff
        self.conj_out = conj_out
        self.deform_interp = deform_interp
        self._cache = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_tm(cls, model, profiles_hd, TM_pix, pola_inout=(2, 2),
                padding_coeff=0.05, conj_out=True, **kwargs):
        """Build by measuring the native geometry from the TM, as ``get_inout_modes`` does.

        The magnification is not assumed: it is the ratio between the RMS width of the
        measured TM marginals and that of the incoherent sum of the theoretical modes.
        """
        profiles_hd = np.asarray(profiles_hd)
        if profiles_hd.ndim == 2:
            n_hd = int(round(np.sqrt(profiles_hd.shape[1])))
            profiles_2d = profiles_hd.reshape(-1, n_hd, n_hd)
        else:
            profiles_2d = profiles_hd

        n2_out, n2_in = TM_pix.shape
        if pola_inout[0] == 2:
            n2_in //= 2
        if pola_inout[1] == 2:
            n2_out //= 2
        n_in, n_out = int(np.sqrt(n2_in)), int(np.sqrt(n2_out))

        w_hd = rms_width(np.mean(np.abs(profiles_2d)**2, axis=0))

        I_in = np.mean(np.abs(TM_pix)**2, axis=0)
        if pola_inout[0] == 2:
            I_in = I_in[:n2_in] + I_in[n2_in:]
        I_out = np.mean(np.abs(TM_pix)**2, axis=1)
        if pola_inout[1] == 2:
            I_out = I_out[:n2_out] + I_out[n2_out:]

        native = {'in': (n_in, rms_width(I_in.reshape(n_in, n_in)) / w_hd),
                  'out': (n_out, rms_width(I_out.reshape(n_out, n_out)) / w_hd)}

        return cls(model, profiles_2d, native, padding_coeff=padding_coeff,
                   conj_out=conj_out, **kwargs)

    # -- core --------------------------------------------------------------

    def ideal_basis(self, side, resolution=None):
        """The theoretical (un-aberrated) basis at `resolution`, same field of view."""
        n_native, zoom = self.native[side]
        resolution = n_native if resolution is None else int(resolution)
        modes = sample_modes(self.profiles, resolution, n_native, zoom)
        if side == 'out' and self.conj_out:
            modes = modes.conj()
        return modes

    def _apply_aberration(self, field, side, n_native):
        """Apply the learned aberration of `side` to a complex tensor (k, N, N)."""
        aber = self.aber[side]
        resolution = field.shape[-1]
        scale = resolution / n_native
        pad, npad, npad_native = _padding(resolution, n_native, self.padding_coeff)

        z = F.pad(field, (pad, pad, pad, pad))

        # global deformation (scaling), relative to the padded array
        deformation = getattr(aber, 'deformation', None)
        theta = getattr(deformation, 'theta', None)
        if theta is not None:
            mask = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32)
            mat = ((1. + theta.detach().cpu()) * mask).reshape(2, 3)
            mat = mat.expand(z.shape[0], 2, 3)
            zr = torch.view_as_real(z).permute(0, 3, 1, 2)
            grid = F.affine_grid(mat.to(zr.dtype), zr.size(), align_corners=True)
            zr = F.grid_sample(zr, grid, mode=self.deform_interp, align_corners=True)
            z = torch.view_as_complex(zr.permute(0, 2, 3, 1).contiguous())

        dtype = z.real.dtype

        # Fourier plane
        z = fft.fftshift(fft.fft2(fft.ifftshift(z, dim=(-2, -1))), dim=(-2, -1))
        ax = _ft_axis(npad, npad_native, scale, dtype=dtype)
        X, Y = torch.meshgrid(ax, ax, indexing='ij')
        for layer in aber.zernike_ft:
            z = z * torch.exp(1j * layer.alpha.detach().to(dtype) * zernike_Z(layer.j, X, Y))

        # direct plane
        z = fft.fftshift(fft.ifft2(fft.ifftshift(z, dim=(-2, -1))), dim=(-2, -1))
        ax = _direct_axis(npad, npad_native, scale, dtype=dtype)
        X, Y = torch.meshgrid(ax, ax, indexing='ij')
        for layer in aber.zernike_direct:
            z = z * torch.exp(1j * layer.alpha.detach().to(dtype) * zernike_Z(layer.j, X, Y))

        start = npad // 2 - resolution // 2
        return z[:, start:start + resolution, start:start + resolution]

    def basis(self, side, resolution=None, normalise=True):
        """The **aberrated** basis at `resolution` -- the change-of-basis matrix.

        Returns
        -------
        array (nmodes, N, N), complex. Row k is mode k as it appears in the modulator
        plane, on a grid of N points spanning the native field of view.
        """
        if side not in ('in', 'out'):
            raise ValueError("side must be 'in' or 'out'")
        n_native, _ = self.native[side]
        resolution = n_native if resolution is None else int(resolution)

        key = (side, resolution, normalise)
        if key in self._cache:
            return self._cache[key]

        ideal = self.ideal_basis(side, resolution)
        with torch.no_grad():
            out = self._apply_aberration(torch.from_numpy(ideal), side, n_native)
        out = out.numpy().astype(complex)
        if normalise:
            out = out / np.linalg.norm(out.reshape(len(out), -1), axis=1).reshape(-1, 1, 1)

        self._cache[key] = out
        return out

    # -- the two operations ------------------------------------------------

    def mask_from_modes(self, coeffs, side, resolution=None, normalise_peak=False):
        """Modulator-plane field for a vector of mode amplitudes.

        ``u = sum_k coeffs[k] * conj(basis[k])``, the convention that matches the
        ``TM_modes = modes_out @ TM_pix @ modes_in^H`` cost the model was fitted with.

        Parameters
        ----------
        coeffs : array (nmodes,)
        side : {'in', 'out'}
        resolution : int, optional
            Grid size. Defaults to the native one. The grid always spans the same
            physical area, so a larger value samples the same field more finely.
        normalise_peak : bool
            Divide by the peak modulus, handy before Lee-hologram encoding.

        Returns
        -------
        array (N, N), complex.
        """
        coeffs = np.asarray(coeffs).ravel()
        B = self.basis(side, resolution)
        if coeffs.size != len(B):
            raise ValueError(f'expected {len(B)} coefficients, got {coeffs.size}')
        field = np.tensordot(coeffs, B.conj(), axes=(0, 0))
        if normalise_peak:
            field = field / (np.abs(field).max() + 1e-30)
        return field

    def modes_from_mask(self, field, side, method='conj'):
        """Mode amplitudes of a modulator-plane field. Inverse of `mask_from_modes`.

        Parameters
        ----------
        field : array (N, N) or (N*N,)
        side : {'in', 'out'}
        method : {'conj', 'pinv'}
            'conj' uses ``a = basis @ u``, exact when the basis rows are orthonormal.
            'pinv' uses the dual basis, exact even when they are not, at the cost of a
            pseudo-inverse.
        """
        field = np.asarray(field)
        resolution = field.shape[-1] if field.ndim == 2 else int(round(np.sqrt(field.size)))
        B = self.basis(side, resolution).reshape(self.nmodes, -1)
        u = field.reshape(-1)
        if method == 'conj':
            return B @ u
        elif method == 'pinv':
            return np.linalg.pinv(B.conj().T) @ u
        raise ValueError("method must be 'conj' or 'pinv'")

    # -- convenience -------------------------------------------------------

    def mask_for_mode(self, k, side, resolution=None, **kwargs):
        """Field that excites a single mode `k`."""
        c = np.zeros(self.nmodes, dtype=complex)
        c[k] = 1.
        return self.mask_from_modes(c, side, resolution, **kwargs)

    def info(self, side, resolution=None):
        """Geometry of a given request, for checking."""
        n_native, zoom = self.native[side]
        resolution = n_native if resolution is None else int(resolution)
        pad, npad, npad_native = _padding(resolution, n_native, self.padding_coeff)
        return {'side': side, 'native_resolution': n_native, 'resolution': resolution,
                'oversampling': resolution / n_native, 'hd_zoom': zoom,
                'padded': npad, 'padded_native': npad_native,
                'hd_pixels_per_output_pixel': (n_native / zoom) / resolution}
