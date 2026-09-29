# Changelog

## [0.2.0] - 2026-09-29
### Fixed
- Replaced AQI proxy (`PM2.5 × 1.5`) with official CPCB breakpoint formula
- Added `pyarrow` dependency for Parquet reads
- Fixed PowerShell artifact at top of `prepare_real_data.py`

### Added
- `prepare_real_data.py` for Vonter Parquet → daily CSV pipeline
- Documentation for AQI methodology and real-world testing results

## [0.1.0] - 2026-09-20
### Added
- Initial classifier pipeline
- Chronological train/test split
- Persistence baseline comparison
- Synthetic data generator
