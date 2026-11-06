# Part 1 Missing Tasks & Implementation Checklist (`todo.md`)

This checklist summarizes the discrepancies between [`part1_expplan.md`](file:///home/luis/Desktop/iacv_labs/assignment1/part1_expplan.md) and the current implementation in [`src/part1_main.py`](file:///home/luis/Desktop/iacv_labs/assignment1/src/part1_main.py) and related modules.

---

## 1. Pipeline & Methods (Table 1 Completion)

- [x] **Add missing binary descriptors for FAST (Section 2.5 & Table 1):**
  - Add `FAST+BRIEF` to experiment dictionary `E` using `descriptors.compute_brief_descriptors`.
  - Add `FAST+FREAK` to experiment dictionary `E` using `descriptors.compute_freak_descriptors`.
  - Ensure keypoint sizes for FAST keypoints are explicitly assigned (e.g. `kp.size = 31.0` matching patch size as in `config.py`) before passing to BRISK, FREAK, or BRIEF.
- [ ] **Record Model Checkpoints (Section 2.6):**
  - Document and save the model checkpoint / weights version used by `SuperPoint` and `LightGlue` in the metadata or configuration log.

---

## 2. Matching Protocol Adherence

- [ ] **Fix Classical Matching Pipeline in `classic_match`:**
  - Replace `cv2.BFMatcher(..., crossCheck=True).match(...)` with the required A1 protocol:
    1. `knnMatch(k=2)`
    2. Lowe's ratio test (ratio < 0.8)
    3. One-to-one enforcement (keep lowest-distance match for each target keypoint)
  - Re-use [`matching.match_pipeline(desc1, desc2, norm_type, ratio=0.8)`](file:///home/luis/Desktop/iacv_labs/assignment1/src/matching.py#L59-L64) from `src/matching.py`.
- [x] **Unify Matching Protocol Across Methods:**
  - Use `cv2.NORM_L2` for SIFT, KAZE, SuperPoint.
  - Use `cv2.NORM_HAMMING` for ORB, FAST+BRIEF, FAST+BRISK, FAST+FREAK.
  - Keep LightGlue as its own learned matcher.

---

## 3. Homography Quality & RANSAC Metrics (Table 3)

- [x] **Estimate Homography via RANSAC per Pair:**
  - Call [`matching.estimate_homography(matches, kps1, kps2)`](file:///home/luis/Desktop/iacv_labs/assignment1/src/matching.py#L108-L130) using `cv2.RANSAC` (threshold 3 px, `maxIters=2000`, `confidence=0.995`).
- [x] **Calculate Homography Accuracy (HA@t):**
  - Project the 4 corners of Image 1 through estimated and ground-truth homographies with [`evaluation.homography_accuracy`](file:///home/luis/Desktop/iacv_labs/assignment1/src/evaluation.py#L326-L351).
  - Record boolean flags for mean corner error $\le 1$ px, $\le 3$ px, and $\le 5$ px (`HA@1px`, `HA@3px`, `HA@5px`).
- [x] **Compute Inlier Statistics:**
  - Compute number of RANSAC inliers and mean inlier reprojection error using [`evaluation.compute_inlier_metrics`](file:///home/luis/Desktop/iacv_labs/assignment1/src/evaluation.py#L357-L392) or [`matching.inlier_stats`](file:///home/luis/Desktop/iacv_labs/assignment1/src/matching.py#L132-L153).

---

## 4. Runtime & Cost Metrics (Table 2)

- [ ] **Convert and Consolidate Stage Runtimes into Milliseconds (`ms`):**
  - Compute `Detect (ms)`, `Describe (ms)`, `Match (ms)`, and `Total (ms)`.
  - For SuperPoint / LightGlue, clearly record `Detect+Describe (ms)` or document joint execution.
- [ ] **Record Descriptor Storage & Hardware Device:**
  - Log `Descriptor dim` (e.g. 128 for SIFT, 64 for KAZE, 256 for ORB/BRIEF, 512 for BRISK/FREAK, 256 for SuperPoint).
  - Log `Bytes/desc.` (512 for SIFT, 256 for KAZE, 32 for ORB/BRIEF, 64 for BRISK/FREAK, 1024 for SuperPoint).
  - Compute `Memory (KB)` $= \frac{\#\text{features} \times \text{bytes per descriptor}}{1024}$.
  - Log `Device` (`CPU` vs `GPU`), especially for comparing SuperPoint and LightGlue fairly against CPU baselines.

---

## 5. Evaluation Setup & Result Aggregation

- [x] **Correct Image Pair Difficulty Ordering:**
  - Change pairs from consecutive `[(1,2), (2,3), (3,4), (4,5), (5,6)]` to standard difficulty reference pairs:
    `PAIRS = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]` (corresponding to $k \in \{2, 3, 4, 5, 6\}$).
- [x] **Split Results by Category (`i` vs `v`):**
  - Tag each row in pair results with `category` (`i` for illumination, `v` for viewpoint) and difficulty `k`.
  - Separate all three summary tables into an `i` block and a `v` block.
- [x] **Generate the 3 Final Report Tables:**
  - **Table 1 (Matching quality):** `#features`, `Repeat.`, `#putative`, `PMR`, `#correct`, `Precision`, `Matching score`, `#corresp.`, `Recall`.
  - **Table 2 (Runtime and cost):** `Detect (ms)`, `Describe (ms)`, `Match (ms)`, `Total (ms)`, `Descriptor dim`, `Bytes/desc.`, `Memory (KB)`, `Device`.
  - **Table 3 (Homography quality):** `HA@1px (%)`, `HA@3px (%)`, `HA@5px (%)`, `RANSAC inliers`, `Mean inlier error (px)`.

---

## 6. FAST Parameter Investigation (Section 2.3)

- [x] **Implement FAST Parameter Sweep Script/Function:**
  - Sweep `threshold` $\in \{5, 10, 15, 20, 30, 40, 60\}$ and `nonmaxSuppression` $\in \{\text{True}, \text{False}\}$.
  - Compare against SIFT baseline on:
    - Number of detected/covisible keypoints
    - Repeatability (one-to-one, 3 px threshold)
    - Detection time (ms per frame)
  - Generate the summary table / curve of Repeatability vs Number of Keypoints and Runtime.

---

## 7. Visualizations & Qualitative Deliverables (Section 5)

- [x] **Keypoint Distribution Panels:**
  - Use [`visualization.draw_keypoints_panel`](file:///home/luis/Desktop/iacv_labs/assignment1/src/visualization.py#L154-L210) to display keypoints across methods on representative images.
- [x] **Correct vs Incorrect Match Figures:**
  - Use [`visualization.draw_classified_matches`](file:///home/luis/Desktop/iacv_labs/assignment1/src/visualization.py#L254-L350) to plot matches on sample pairs (e.g., 1 illumination pair and 1 viewpoint pair) with green lines for correct matches ($\le 3$ px) and red lines for incorrect matches.
- [x] **Degradation & Pareto Plots:**
  - Call [`visualization.plot_degradation`](file:///home/luis/Desktop/iacv_labs/assignment1/src/visualization.py#L407-L482) for `Matching_Score` vs $k$ across `i` and `v`.
  - Call [`visualization.plot_pareto`](file:///home/luis/Desktop/iacv_labs/assignment1/src/visualization.py#L516-L600) for runtime vs `Matching_Score` / `HA@3px`.
