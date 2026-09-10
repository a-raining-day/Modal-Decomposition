# Memory-grid summary

Each cell: fresh subprocess; decomposition budget counted after the input was fully built. 'deferred' cells were recorded but not executed (see POSTPONED_3GB.md).

## MD-EMD

| size | pattern | status | wall (s) | peak RSS (MB) | rows | note |
|----:|---------|--------|---------:|--------------:|-----:|------|
| 500MB | increasing | ok | 4.313 | 4576.0 | 1 |  |
| 500MB | random | timeout |  | 7788.6 |  | killed after decompose budget |
| 1024MB | increasing | ok | 10.797 | 8119.4 | 1 |  |
| 1024MB | random | deferred |  | 0.0 |  | random noise at >= 512MB is postponed on this 16GB machine -... |
| 1MB | increasing | ok | 0.0 | 113.0 | 1 |  |
| 1MB | random | ok | 10.656 | 141.6 | 17 |  |
| 20MB | increasing | ok | 0.125 | 273.8 | 1 |  |
| 20MB | random | timeout |  | 628.3 |  | killed after decompose budget |
| 100MB | increasing | ok | 0.734 | 992.2 | 1 |  |
| 100MB | random | timeout |  | 2669.7 |  | killed after decompose budget |

## PyEMD

| size | pattern | status | wall (s) | peak RSS (MB) | rows | note |
|----:|---------|--------|---------:|--------------:|-----:|------|
| 500MB | increasing | ok | 3.391 | 4100.3 | 1 |  |
| 500MB | random | timeout |  | 8575.7 |  | killed after decompose budget |
| 1024MB | increasing | ok | 9.016 | 8289.6 | 1 |  |
| 1024MB | random | deferred |  | 0.0 |  | random noise at >= 512MB is postponed on this 16GB machine -... |
| 1MB | increasing | ok | 0.016 | 113.3 | 1 |  |
| 1MB | random | ok | 10.75 | 138.9 | 17 |  |
| 20MB | increasing | ok | 0.125 | 234.7 | 1 |  |
| 20MB | random | timeout |  | 608.3 |  | killed after decompose budget |
| 100MB | increasing | ok | 0.578 | 816.9 | 1 |  |
| 100MB | random | timeout |  | 2572.3 |  | killed after decompose budget |

## PySDKit

| size | pattern | status | wall (s) | peak RSS (MB) | rows | note |
|----:|---------|--------|---------:|--------------:|-----:|------|
| 500MB | increasing | ok | 3.844 | 4058.2 | 1 |  |
| 500MB | random | timeout |  | 8000.9 |  | killed after decompose budget |
| 1024MB | increasing | ok | 9.812 | 8160.5 | 1 |  |
| 1024MB | random | deferred |  | 0.0 |  | random noise at >= 512MB is postponed on this 16GB machine -... |
| 1MB | increasing | ok | 0.0 | 119.0 | 1 |  |
| 1MB | random | ok | 11.422 | 158.9 | 17 |  |
| 20MB | increasing | ok | 0.109 | 250.7 | 1 |  |
| 20MB | random | timeout |  | 611.9 |  | killed after decompose budget |
| 100MB | increasing | ok | 0.593 | 818.2 | 1 |  |
| 100MB | random | timeout |  | 2566.7 |  | killed after decompose budget |
