# Benchmark Report Tables

Aggregated quantitative evaluation on HPatches dataset across Illumination (i_*) and Viewpoint (v_*) sequences.

### Table 1: Matching Quality

| Category | Method | #features | Repeat. | #putative | PMR | #correct | Precision | Matching score | #corresp. | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Illumination | SIFT | 790.9 | 0.235 | 116.1 | 0.139 | 81.0 | 0.490 | 0.086 | 188.3 | 0.327 |
| Illumination | ORB | 995.4 | 0.313 | 87.9 | 0.088 | 67.5 | 0.454 | 0.068 | 311.9 | 0.170 |
| Illumination | FAST_BRISK | 1095.7 | 0.353 | 113.5 | 0.104 | 97.1 | 0.576 | 0.084 | 350.6 | 0.219 |
| Illumination | FAST_BRIEF | 1197.4 | 0.336 | 184.0 | 0.158 | 155.7 | 0.589 | 0.126 | 372.3 | 0.326 |
| Illumination | FAST_FREAK | 857.9 | 0.413 | 106.4 | 0.121 | 83.0 | 0.474 | 0.092 | 301.6 | 0.218 |
| Illumination | KAZE | 556.2 | 0.386 | 118.5 | 0.194 | 100.4 | 0.662 | 0.162 | 219.1 | 0.323 |
| Illumination | SuperPoint_BF | 1017.8 | 0.542 | 258.7 | 0.237 | 228.3 | 0.811 | 0.208 | 555.8 | 0.355 |
| Illumination | SuperPoint_LightGlue | 1017.8 | 0.542 | 595.8 | 0.579 | 506.7 | 0.845 | 0.492 | 555.8 | 0.907 |
| Viewpoint | SIFT | 702.2 | 0.545 | 277.2 | 0.407 | 227.6 | 0.775 | 0.340 | 362.3 | 0.584 |
| Viewpoint | ORB | 863.7 | 0.548 | 194.0 | 0.216 | 164.9 | 0.790 | 0.183 | 478.7 | 0.317 |
| Viewpoint | FAST_BRISK | 911.9 | 0.580 | 142.5 | 0.152 | 114.6 | 0.652 | 0.117 | 543.3 | 0.193 |
| Viewpoint | FAST_BRIEF | 973.9 | 0.623 | 264.9 | 0.283 | 222.0 | 0.739 | 0.227 | 598.7 | 0.351 |
| Viewpoint | FAST_FREAK | 718.3 | 0.529 | 137.8 | 0.172 | 109.2 | 0.682 | 0.134 | 406.4 | 0.242 |
| Viewpoint | KAZE | 1090.8 | 0.550 | 323.7 | 0.304 | 254.6 | 0.795 | 0.252 | 612.4 | 0.438 |
| Viewpoint | SuperPoint_BF | 1191.6 | 0.749 | 632.0 | 0.484 | 547.9 | 0.872 | 0.425 | 839.9 | 0.581 |
| Viewpoint | SuperPoint_LightGlue | 1191.6 | 0.749 | 907.2 | 0.772 | 802.3 | 0.904 | 0.702 | 839.9 | 0.943 |

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

### Table 3: Homography Estimation Quality

| Category | Method | HA@1px (%) | HA@3px (%) | HA@5px (%) | RANSAC inliers | Mean inlier error (px) |
| --- | --- | --- | --- | --- | --- | --- |
| Illumination | SIFT | 20.0 | 40.0 | 50.0 | 81.2 | 0.71 |
| Illumination | ORB | 0.0 | 20.0 | 30.0 | 68.4 | 0.72 |
| Illumination | FAST_BRISK | 30.0 | 50.0 | 50.0 | 97.5 | 0.58 |
| Illumination | FAST_BRIEF | 30.0 | 30.0 | 50.0 | 153.2 | 0.79 |
| Illumination | FAST_FREAK | 30.0 | 30.0 | 30.0 | 84.2 | 0.57 |
| Illumination | KAZE | 20.0 | 50.0 | 60.0 | 101.4 | 0.61 |
| Illumination | SuperPoint_BF | 30.0 | 60.0 | 80.0 | 229.1 | 1.10 |
| Illumination | SuperPoint_LightGlue | 20.0 | 80.0 | 90.0 | 505.1 | 1.34 |
| Viewpoint | SIFT | 30.0 | 60.0 | 80.0 | 232.8 | 0.98 |
| Viewpoint | ORB | 0.0 | 30.0 | 60.0 | 167.8 | 1.18 |
| Viewpoint | FAST_BRISK | 10.0 | 20.0 | 20.0 | 114.9 | 0.98 |
| Viewpoint | FAST_BRIEF | 0.0 | 50.0 | 90.0 | 223.9 | 1.01 |
| Viewpoint | FAST_FREAK | 10.0 | 30.0 | 40.0 | 111.2 | 0.93 |
| Viewpoint | KAZE | 20.0 | 70.0 | 100.0 | 258.3 | 1.21 |
| Viewpoint | SuperPoint_BF | 20.0 | 80.0 | 100.0 | 549.8 | 1.25 |
| Viewpoint | SuperPoint_LightGlue | 30.0 | 80.0 | 100.0 | 790.6 | 1.30 |
