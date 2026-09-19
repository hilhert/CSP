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