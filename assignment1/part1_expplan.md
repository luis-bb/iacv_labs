Part 1 (sections 2.1 to 2.8) is a comparison of local feature methods, and its deliverable is quantitative evidence, not displayed keypoints. In the assignment’s words, “the objective of Part 1 is not simply to display detected points.”

Required

SIFT is the main classical baseline, evaluated on both detection and matching (2.2).


FAST: investigate the intensity threshold and non-maximum suppression, and compare against SIFT on repeatability, number of points and runtime (2.3).


ORB and KAZE: analyse the trade-off between speed, descriptor size, robustness and matching accuracy (2.4).


FAST with a binary descriptor (BRIEF, BRISK or FREAK, “where available”). If one is missing in your OpenCV build, say so explicitly (2.5).


SuperPoint + descriptor matching and SuperPoint + LightGlue, using a documented pretrained implementation, with the checkpoint recorded (2.6, section 4).
Quantitative evaluation (2.7, 2.8):
Data is HPatches, with results reported separately for at least two categories (illumination and viewpoint).
Measurements are keypoint count, repeatability, correct matches, homography accuracy, RANSAC inliers, reprojection error, runtime, and descriptor size and memory.
The four named metrics are PMR, precision, matching score and recall, with counts and denominators. A match is correct within 3 px, and the ratio test is 0.8 or your own stated threshold.
Results and analysis (section 5): one table of repeatability and matching, one of runtime and cost, keypoint visualisations, correct and incorrect match examples, and discussions of robustness (scale, rotation, illumination, viewpoint), accuracy vs efficiency, and real-time vs accuracy-first use.
The report must keep detector, descriptor and matcher performance conceptually separate.

Matching is a minor part of Part 1, and `cv2.BFMatcher` is what the assignment recommends (section 11: "Use cv2.BFMatcher or an appropriate OpenCV matcher").

**What to do**
- `cv2.BFMatcher(NORM_L2)` for SIFT, KAZE and SuperPoint.
- `cv2.BFMatcher(NORM_HAMMING)` for ORB, BRIEF, BRISK and FREAK.
- `knnMatch(k=2)`, keep matches with ratio < 0.8, then enforce one-to-one by keeping only the lowest-distance match for each target keypoint.
- Use the same settings for every method, and state them in the report.

**One catch:** `crossCheck=True` only works with `k=1`, so it can't be combined with the ratio test. Do the one-to-one step yourself, as above.

**Exception:** SuperPoint + LightGlue uses LightGlue as its matcher, so BFMatcher doesn't apply there.

# what tables should result of part 1?

Make three tables, each with one row per method. Every value is the mean over pairs, and each table has an `i` block and a `v` block (repeat the table, or put them side by side).

**Table 1: matching quality**

| Method | #features | Repeat. | #putative | PMR | #correct | Precision | Matching score | #corresp. | Recall |
|---|---|---|---|---|---|---|---|---|---|
| SIFT | | | | | | | | | |
| KAZE | | | | | | | | | |
| ORB | | | | | | | | | |
| FAST+BRIEF | | | | | | | | | |
| FAST+BRISK | | | | | | | | | |
| FAST+FREAK | | | | | | | | | |
| SuperPoint + BF | | | | | | | | | |
| SuperPoint + LightGlue | | | | | | | | | |

- Mean counts (#features, #putative, #correct, #corresp.) next to the ratios, so the metrics can be reproduced, as section 2.8 requires.
- #corresp. is the recall denominator.

**Table 2: runtime and cost**

| Method | Detect (ms) | Describe (ms) | Match (ms) | Total (ms) | Descriptor dim | Bytes/desc. | Memory (KB) | Device |
|---|---|---|---|---|---|---|---|---|

- Device = CPU or GPU, so SuperPoint and LightGlue can be compared fairly.
- Memory = #features × bytes per descriptor.

**Table 3: homography quality**

| Method | HA@1px (%) | HA@3px (%) | HA@5px (%) | RANSAC inliers | Mean inlier error (px) |
|---|---|---|---|---|---|

- HA@t = percentage of pairs where the mean corner error is ≤ t px.

The assignment's section 5 also asks for one table on repeatability/matching, one on runtime/cost, and homography quality, so these three cover it.

**Optional extra:** a short table of FAST parameters (threshold, NMS on/off against #keypoints, repeatability and time), since section 2.3 asks you to study them.