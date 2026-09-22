## [v0.2-mod3] — 2026-09-19

### Added
- CSP-Fast 5-seed length generalization results (parity, mod3, Dyck-1)
- Per-seed accuracy tables
- Baseline comparison tables (single-seed, to be replaced)

### Changed
- README TL;DR: clarified that 100% refers to the training length, not all evaluated lengths
- Architecture section: corrected CSP-Fast description (accumulated-phase rotation, not associative scan)
- Length generalization tables now report mean ± std for CSP-Fast

### Known limitations
- Baseline seed runs pending
- Length generalization figures not yet regenerated from aligned runs
- Mamba negative-eigenvalue variant casually shows NaN collapse after convergence (to be fixed in next round)



## [v0.3-atan] — 2026-09-22

### Added
- Full 5-seed length generalization results for all six models across
  parity, mod-3, and Dyck-1 (mean ± std reported)
- Baseline comparison tables with seed variance for Mamba and Transformer
- Mod-3 length generalization curves showing CSP-Vanilla and CSP-Fast
  outperform Mamba and the Transformer at L=64

### Changed
- Rotation projection changed from `tanh(·) * π` to `atan2(·)`. The
  `atan2` form extracts the phase directly from a 2D projection, removing
  the artificial magnitude bound imposed by `tanh` and simplifying the
  parametrization. The magnitude of the projection is now irrelevant to
  the extracted phase.
- CSP-Vanilla on mod-3: L=64 accuracy improved from 0.537 (previous run)
  to 0.921 (mean across seeds), with one seed reaching 1.000.
- CSP-Fast on mod-3: L=64 accuracy improved from 0.594 to 0.876, with one
  seed reaching 1.000.
- README results tables regenerated with the current code and seeds.
- Training table updated: epochs to 100% accuracy reduced to ~10 for
  all tasks after the atan2 change.

### Fixed
- Removed the `tanh(·) * π` magnitude bounding in the rotation projection,
  which was suppressing the tail of the phase distribution and preventing
  the model from reaching phase values near ±π.
- README architecture section now correctly describes the phase extraction
  as `atan2(W_θ z_t)` rather than the previous cumsum-of-tanh formulation.

### Known limitations
- Mod-3 seed variance remains high for CSP (CSP-Vanilla std = 0.145,
  CSP-Fast std = 0.171). The exact-phase solution is reachable by gradient
  descent but not reliably found across random initializations.
- CSP does not reach the exact-transition tier on parity. Vanilla RNN and
  Complex RNN maintain accuracy ≥ 0.98 at all evaluation lengths.
- The Dyck-1 task favors the Transformer, which achieves 0.983 at L=64.
  CSP-Fast reaches 0.919.
- Mamba negative-eigenvalue variant shows occasional NaN collapse under
  some configurations; a bounded eigenvalue parametrization is
  recommended for stable training.

### Baseline seeds
- All baseline results (Vanilla RNN, Complex RNN, Mamba, Transformer)
  are now averaged across five seeds with the same protocol as CSP.
- No data was excluded or selected; the reported values are the full
  mean ± std over seeds 42–46.

