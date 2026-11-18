# AgiBot-World #164 pre-fix baseline (frozen)

Baseline for the fix-anchor protocol proposed by @PHM-Audit on
[OpenDriveLab/AgiBot-World#164](https://github.com/OpenRAL/openral/pull/451)/[#164](https://github.com/OpenDriveLab/AgiBot-World/issues/164):
**7/7 pre-fix → 0/7 post-fix** is the closure criterion.

## Frozen artifacts

- `prefix-sweep.json` — per-task validator output (2026-11-17 sweep snapshot)
- rule-set: embodied-lake v0.1.3 `quaternion_nonunit` (calibrated 2026-10-28,
  see CHANGELOG 0.1.2) + frozen/saturation rules
- thresholds: components ≤ 1.5 · norm median ∈ [0.95, 1.6] · CV < 0.2 ·
  deviation > 1e-3 in > 50% of frames

## Baseline expectations (per task, per channel)

| task | channels hit (norm median) | deviating frames |
|---|---|---|
| 676  | ee_base (1.007) · ee_camera (1.248) | 663/751 · 737/751 |
| 705  | dex (1.115) | 786/786 |
| 604  | ee_base (1.006) · ee_camera (1.316) | 5,380/7,474 · 7,474/7,474 |
| 727  | ee_base (1.050) · ee_camera (1.038) | 7,670/8,696 · 8,476/8,696 |
| 568  | ee_camera (1.236) | 2,277/2,277 |
| 719  | ee_base (1.007) · ee_camera (1.274) | 9,795/11,130 · 11,130/11,130 |
| 737  | ee_camera (1.277) | 24,034/24,034 |
| 619  | ee_base (1.004) · ee_camera (1.185) | 29,904/36,080 · 36,080/36,080 |
| 429  | ee_base (1.002) · ee_camera (1.324) | 14,828/25,758 · 25,758/25,758 |

Unscanned (not clean): 730, 446, 677 — see failure records in
`prefix-sweep.json` / `prefix-sweep-v2.json` (data sections exceeding the
streaming budget; v2 re-ran 619/429 successfully with a larger cap).
**Running total: 9/9 scanned tasks affected, ~117,000 frames.**

## Post-fix re-score protocol

1. After a fix release of `AgiBotWorld-Beta` tarballs, re-run:
   `python scripts/validate_hub_tarball.py <task>.tar.gz --out /tmp/<task>_post`
2. Same rule-set, same thresholds, same sub-episode frame sets (tabular head
   extraction is deterministic on a tarball's member order).
3. Pass criterion per task, per channel (pre-registered, amended 2026-11-18
   per @PHM-Audit's physical criterion — stricter than "no finding"):
   - **unit band restored**: median |norm − 1| ≤ 1e-3
   - **residual**: 1e-3 < median |norm − 1| ≤ 1e-2 (improved, not fully corrected)
   - **uncorrected**: median |norm − 1| > 1e-2
   A channel hit in the baseline that lands in `residual` or `uncorrected`
   post-fix is a first signal its reference was not fully corrected, even if
   the rule no longer fires. Channels not hit in the baseline are exempt.

Rule-set changes (if any) require re-running the pre-fix baseline on the
pre-fix tarballs for a like-for-like comparison.
