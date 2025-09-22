"""
Module 5_complete: Finalize and decode compressed sets

Maps selected RowIds from compressed/coreset outputs back to the original
encoded CSVs, then decodes to raw_processed format using the pre-fitted
encoders.

Components:
- finalize_decode.py: resolve coreset/test RowIds → write encoded_compressed.csv
  → inverse_transform to raw_processed_compressed.csv

Key features:
- Standardized coreset path resolution for train/test (Benign uses filtered base)
- In-place safe: does not alter inputs; writes new encoded/raw outputs
- Supports batch mode across all major labels (`--label All`)

See README.md for usage and data flow.
"""


