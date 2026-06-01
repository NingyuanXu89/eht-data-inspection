# Data Directory

Use `data/sample/` for small files needed by examples or tests.

Large EHT/VLBI products such as full UVFITS files, HOPS fringe products, and
pipeline-stage alist dumps should normally stay outside version-controlled
package data. Pass their locations explicitly to loaders such as
`eht_inspection.alist.load_alist(..., data_dir=...)`.

