# Release notes

## 0.3

### New features

- Add `PyTorchAberrations/mode_masks.py`: given a fitted model and a vector of mode
  amplitudes, build the field to display in the modulator plane on a grid of **any**
  size, spanning the same physical area as the pixel basis the model was fitted on.
  `ModeMasks.mask_from_modes` / `.modes_from_mask` / `.basis`, with `side='in'` or
  `'out'` selecting the relevant aberration. Two subtleties are handled, and both are
  invisible until measured:
  - the basis is sampled on the grid `scaling_functions.resize_modes` actually lays
    down, whose pitch is `(n_hd - 1) / (n_zoom - 1)` rather than the nominal `1 / zoom`
    and whose origin carries a floor-divided crop offset. Using a nominally clean centred
    grid instead gives a *different physical field*, 56 % away from the one the model was
    fitted on;
  - the direct-plane Zernike coordinates keep their physical meaning across resolutions
    while the Fourier-plane ones cover a Nyquist band `s` times wider, so the pupil
    occupies `1/s` of the normalised domain. Ignoring that costs 69 %.

  At the native resolution the module reproduces the fitted change-of-basis matrix to
  seven digits, which is the check `examples/Demo_mode_masks.ipynb` opens with.
- Add `examples/Demo_mode_masks.ipynb`, which loads a saved fit — no optimisation — and
  compares the reference low-resolution field against the model at the same resolution
  and at three times finer.
- `examples/Demo_correction_aberration.ipynb` now saves the fitted weights *and* the
  corrected change-of-basis matrices to `examples/data/fitted_correction.pt`, for the
  notebook above. The file is generated, not tracked (see `.gitignore`).
- Add `tests/test_mode_masks.py`.

### Bug correction

- `examples/Demo_correction_aberration.ipynb` reported the change-of-basis matrices from
  the forward pass *preceding* the last `optimizer.step()`, so they were one Adam step
  behind the model. Harmless for the published figures, but it made them inconsistent
  with the saved weights by 0.5 %.

### Changes

- **The deformation layers now interpolate bicubically** rather than bilinearly.
  `ComplexScaling` and `ComplexDeformation` resample the field at sub-pixel positions,
  and a straight line through two neighbours is a poor fit to a mode sampled at barely
  three points per oscillation: shifting a mode by half a pixel that way loses 18 % of
  it, against 3 % with a cubic through four. `mode='bicubic'` is differentiable in
  current PyTorch — it was not when the model was first written — so the change is free.
  On the bundled dataset it takes the fit from

  | | conversion | diagonal |
  |---|---|---|
  | bilinear | 92.38 % | 89.12 % |
  | bicubic | 92.55 % | **91.23 %** |

  in the same 30 s. The block-diagonal ratio is not optimised by anything, so its
  2.1-point gain is independent evidence. Two side effects worth knowing: the fitted
  scaling changes (`theta` goes from -0.0457 to -0.0515 — it was partly absorbing the
  interpolation error), so **coefficients fitted with an earlier version no longer mean
  the same thing**; and the corrected basis comes out closer to orthonormal, which
  quarters the error of `modes_from_mask(..., method='conj')`.
- `example/` renamed to `examples/`, now that there is more than one.


## 0.2

### New features

- Add `pyproject.toml`, the package is now installable with `pip install -e .`
- Add unit tests (`tests/`), run with `pytest`
- Add `PyTorchAberrations/cost_functions.py` with `norm_mode_to_norm_pix` and
  `energy_on_diagonal`
- Add `PyTorchAberrations/scaling_functions.py` to rescale a high-definition mode set
  onto the input and output pixel grids, the magnification being measured from the
  transmission matrix rather than assumed
- Add `PyTorchAberrations/plotting_functions.py` (`colorize`, `logplotTM`,
  `plot_outlines`)
- Add sample data in `example/data/` (`TM_pix.npy`, `modes_hd.npz`)
- Rewrite the example notebook: it now tracks the block-diagonal energy during the fit as
  well as the conversion ratio, and states the expected result so a run can be checked

### Bug correction

- **Coma.** `zernike_Z` built `j = 7` and `j = 8` (vertical and horizontal coma) with
  `sin(3*theta)` and `cos(3*theta)`. Coma has azimuthal order `m = ±1`, so the model had
  no coma term at all, while the `m = ±3` sector was covered twice: those two terms
  differed from the trefoils `j = 6` and `j = 9` only by their radial polynomial. The
  redundancy is visible in published fits, where `tref_V`/`coma_V` and `coma_H`/`tref_H`
  come out as near-identical pairs. All 14 polynomials now match the standard OSA/ANSI
  definitions.
  **Warning**: this changes the meaning of any previously fitted `coma_V` / `coma_H`
  coefficient.
- **Identity at initialisation.** `ComplexScaling` and `ComplexDeformation` called
  `affine_grid` with the default `align_corners=False` while `grid_sample` used
  `align_corners=True`. The two conventions differ by half a pixel, so an untrained
  layer resampled its input instead of returning it — a 56 % relative error at a
  35 x 35 grid, meaning every fit started from a displaced basis.
- **`ComplexDeformation` could not run.** `view_as_complex` requires a contiguous last
  dimension, which `permute` does not give, so `deformation='single'` raised
  *"Tensor must have a last dimension with stride 1"*. `ComplexScaling` already had the
  `.contiguous()` call.
- Restore `getZernikeCoefs` and `showZernikeCoefs`, removed along with
  `aberration_functions.py`, which left the last three cells of the demo notebook
  raising `NameError`. `set_xticks` is now called before `set_xticklabels`, as recent
  matplotlib requires to guarantee the labels land on the right ticks.
- The two Zernike coefficient figures in the demo notebook had their titles swapped:
  `best_Zernike_coeff[0]` comes from `abberation_output` but was labelled *Input*.

### Changes

- Compatibility with recent PyTorch: `torch.meshgrid` is now called with an explicit
  `indexing='ij'`, which warns today and is scheduled to become an error
- Repository layout, so that the package is self-contained once installed:
  - `scaling_functions.py` and `plotting_functions.py` moved into `PyTorchAberrations/`
  - the demo notebook and its data moved to `example/` and `example/data/`
  - unit tests in `tests/`
- Known limitation, left unchanged: `ComplexDeformation` multiplies its parameter vector
  by `[1, 0, 0, 0, 1, 0]`, so the two shears and the two shifts have exactly zero
  gradient and cannot be learned, despite the docstring announcing 6 parameters. Only
  the `x` and `y` scalings are active. Unmasking them changes the capacity of the model,
  so it is deliberately left as a decision to make. The test suite records this as an
  expected failure.

## 0.1

First public version, accompanying
[*Learning and avoiding disorder in multimode fibers*](https://arxiv.org/abs/2010.14813),
M. W. Matthès, Y. Bromberg, J. de Rosny and S. M. Popoff,
Phys. Rev. X **11**, 021060 (2021).
