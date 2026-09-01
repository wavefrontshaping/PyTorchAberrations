# PyTorchAberrations

Differentiable aberration layers for PyTorch using Zernike polynomials.

Optical aberrations and misalignments, expressed as PyTorch modules: a global deformation
and Zernike phase screens in the Fourier and direct planes, all with learnable
coefficients. Because they are differentiable, the aberrations of a setup can be *fitted*
rather than calibrated by hand.

The application it was written for is multimode fiber characterisation. A transmission
matrix measured in the pixel basis should, once projected onto the theoretical mode basis
of the fiber, be close to block diagonal. Aberrations spoil that projection; fitting them
recovers it. See
[*Learning and avoiding disorder in multimode fibers*](https://arxiv.org/abs/2010.14813),
M. W. Matthès, Y. Bromberg, J. de Rosny and S. M. Popoff, Phys. Rev. X **11**, 021060
(2021).

## Install

```bash
pip install -e .
```

Requires `torch`, `numpy`, `scipy` and `matplotlib`. Extras: `[test]` for `pytest`,
`[notebooks]` for `jupyter` and `tqdm`.

## Getting started

[`examples/Demo_correction_aberration.ipynb`](examples/Demo_correction_aberration.ipynb)
runs the whole thing on the bundled dataset, in about 30 s on a CPU. On that data the
correction takes the transmission matrix from

| | conversion ratio | diagonal ratio |
|---|---|---|
| before | 60.2 % | 16.4 % |
| after | 92.6 % | 91.2 % |

The *conversion ratio* is the fraction of the pixel-basis TM energy surviving the
projection onto the mode basis, and is what the cost maximises. The *diagonal ratio* is
the fraction of the mode-basis TM energy inside the near-degenerate blocks; nothing
optimises it, so its improvement is independent evidence that the correction is physical.

## Layout

```
PyTorchAberrations/      the package
  aberration_layers.py     Zernike, scaling and deformation layers
  aberration_models.py     Aberration, AberrationModes
  cost_functions.py        norm_mode_to_norm_pix, energy_on_diagonal, normalize
  scaling_functions.py     resample a high-resolution mode set onto the pixel grids
  mode_masks.py            mode amplitudes -> modulator field, at any resolution
  plotting_functions.py    colorize, logplotTM, showZernikeCoefs
examples/                notebooks and their data
tests/                   pytest
```

## Masks at an arbitrary resolution

The fitted change-of-basis matrix lives on the pixel grid the transmission matrix was
measured on — 35x35 and 41x41 for the bundled data. Those are rarely the grids you want
to address a modulator with. Since the fitted model describes the *optics*, it can be
applied to a mode basis sampled as finely as you like:

```python
from PyTorchAberrations.mode_masks import ModeMasks

mm = ModeMasks.from_tm(model, profiles, TM_pix, pola_inout=(2, 2), padding_coeff=0.05)
field = mm.mask_from_modes(coeffs, side='in', resolution=105)   # same area, 3x finer
```

[`examples/Demo_mode_masks.ipynb`](examples/Demo_mode_masks.ipynb) walks through it and
checks the result against the fitted basis. Run
`Demo_correction_aberration.ipynb` first: it writes the fit to
`examples/data/fitted_correction.pt`, which is deliberately not tracked by git.

Two things have to be right for a coefficient fitted on one grid to mean the same
aberration on another — the sampling grid the fit actually used, and the rescaling of the
Fourier-plane coordinates. Getting either wrong is a 56 % and a 69 % error respectively,
on fields that look perfectly plausible. See the `mode_masks` module docstring.

## A note on the scaling step

`scaling_functions.get_inout_modes` runs *before* the model and is easy to overlook. The
theoretical modes arrive on their own grid with no a-priori relation to the modulator and
camera pixels. Rather than assuming a magnification, it measures one — matching the RMS
width of the incoherent mode sum against the marginals of the measured transmission
matrix — and resamples the modes onto both pixel grids.

## Tests

```bash
pip install -e .[test]
pytest
```

Each test guards a bug that once reached this repository: the Zernike table against the
standard OSA/ANSI definitions, the identity of an untrained model across every
deformation mode and padding coefficient, and `ComplexDeformation` running at all.

## Known limitation

`ComplexDeformation` multiplies its parameter vector by `[1, 0, 0, 0, 1, 0]`, so the two
shears and the two shifts have exactly zero gradient and cannot be learned, despite the
docstring announcing six parameters. Only the x and y scalings are active. Unmasking them
changes the capacity of the model, so it is left as a deliberate decision; the test suite
records it as an expected failure.

## Release notes

See [CHANGES.md](CHANGES.md).
