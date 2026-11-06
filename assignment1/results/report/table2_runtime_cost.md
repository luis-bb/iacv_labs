### Table 2: Runtime and Computational Cost

| Category | Method | Detect (ms) | Describe (ms) | Match (ms) | Total (ms) | Descriptor dim | Bytes/desc. | Memory (KB) | Device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Illumination | SIFT | 38.65 | 26.21 | 4.64 | 134.36 | 128 | 512 | 447.7 | CPU |
| Illumination | ORB | 5.13 | 3.92 | 2.45 | 20.55 | 256 | 32 | 31.2 | CPU |
| Illumination | FAST_BRISK | 1.97 | 19.65 | 4.38 | 47.62 | 512 | 64 | 72.3 | CPU |
| Illumination | FAST_BRIEF | 1.98 | 2.58 | 4.67 | 13.79 | 256 | 32 | 38.9 | CPU |
| Illumination | FAST_FREAK | 1.97 | 19.61 | 3.24 | 46.41 | 512 | 64 | 60.4 | CPU |
| Illumination | KAZE | 379.63 | 291.72 | 4.55 | 1347.24 | 64 | 256 | 209.3 | CPU |
| Illumination | SuperPoint_BF | 1646.58 | 0.00 | 22.77 | 3315.92 | 256 | 1024 | 1205.6 | CPU |
| Illumination | SuperPoint_LightGlue | 1699.65 | 0.00 | 502.95 | 3902.26 | 256 | 1024 | 1205.6 | CPU |
| Viewpoint | SIFT | 29.02 | 24.34 | 4.83 | 111.54 | 128 | 512 | 445.3 | CPU |
| Viewpoint | ORB | 5.02 | 3.81 | 2.27 | 19.93 | 256 | 32 | 29.2 | CPU |
| Viewpoint | FAST_BRISK | 1.38 | 18.80 | 3.87 | 44.23 | 512 | 64 | 65.7 | CPU |
| Viewpoint | FAST_BRIEF | 1.47 | 2.47 | 4.34 | 12.21 | 256 | 32 | 35.7 | CPU |
| Viewpoint | FAST_FREAK | 1.65 | 22.68 | 3.08 | 51.75 | 512 | 64 | 51.4 | CPU |
| Viewpoint | KAZE | 355.18 | 292.47 | 15.37 | 1310.66 | 64 | 256 | 330.1 | CPU |
| Viewpoint | SuperPoint_BF | 1738.07 | 0.00 | 36.76 | 3512.90 | 256 | 1024 | 1550.3 | CPU |
| Viewpoint | SuperPoint_LightGlue | 1810.51 | 0.00 | 612.87 | 4233.89 | 256 | 1024 | 1550.3 | CPU |
