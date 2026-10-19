from __future__ import annotations

import math

from agents.types import MatchResult


class EvaluatorAgent:
    """Scores and ranks matching results with geometric quality signals."""

    def __init__(
        self,
        objective: str = "综合平衡",
        custom_weights: dict[str, float] | None = None,
    ) -> None:
        self.objective = objective
        self.custom_weights = self._normalize_weights(custom_weights)

    def score(self, results: list[MatchResult]) -> list[MatchResult]:
        valid_results = [result for result in results if result.error is None]
        if not valid_results:
            return []

        max_inliers = max((result.num_inliers for result in valid_results), default=1) or 1
        max_matches = max((result.num_good_matches for result in valid_results), default=1) or 1
        max_time = max((result.runtime_ms for result in valid_results), default=1.0) or 1.0

        for result in valid_results:
            inlier_count_score = result.num_inliers / max_inliers
            match_count_score = result.num_good_matches / max_matches
            speed_score = 1.0 - min(result.runtime_ms / max_time, 1.0)
            reprojection_score = self._reprojection_score(result.reprojection_error)
            geometry_score = result.homography_quality
            coverage_score = self._coverage_score(result.match_coverage)
            strictness_score = self._strictness_score(result)
            evidence_score = result.evidence_score
            spatial_score = result.spatial_distribution_score

            weights = self._weights()
            result.score = (
                weights["inlier_ratio"] * result.inlier_ratio
                + weights["inlier_count"] * inlier_count_score
                + weights["match_count"] * match_count_score
                + weights["reprojection"] * reprojection_score
                + weights["geometry"] * geometry_score
                + weights["coverage"] * coverage_score
                + weights["evidence"] * evidence_score
                + weights["spatial_distribution"] * spatial_score
                + weights["speed"] * speed_score
                + weights["strictness"] * strictness_score
            )
            result.score_breakdown = {
                "inlier_ratio": result.inlier_ratio,
                "inlier_count": inlier_count_score,
                "match_count": match_count_score,
                "reprojection": reprojection_score,
                "geometry": geometry_score,
                "coverage": coverage_score,
                "evidence": evidence_score,
                "spatial_distribution": spatial_score,
                "speed": speed_score,
                "strictness": strictness_score,
            }

            penalty_scale = self._penalty_scale()
            if result.num_inliers < 4:
                result.score -= 0.2
            elif result.num_inliers < 8:
                result.score -= 0.18 * penalty_scale
            elif result.num_inliers < 12:
                result.score -= 0.08 * penalty_scale
            if result.homography is None:
                result.score -= 0.15
            if result.reprojection_error is not None and result.reprojection_error > 8:
                result.score -= 0.05
            if result.homography_quality <= 0.05:
                result.score -= 0.65 * penalty_scale
            elif result.homography_quality < 0.30:
                result.score -= 0.32 * penalty_scale
            if result.projected_inside_ratio < 0.50:
                result.score -= 0.25 * penalty_scale
            if result.inlier_ratio < 0.25:
                result.score -= 0.08 * penalty_scale
            elif result.inlier_ratio < 0.35:
                result.score -= 0.03 * penalty_scale
            if result.match_coverage < 0.15:
                result.score -= 0.45 * penalty_scale
            elif result.match_coverage < 0.30:
                result.score -= 0.24 * penalty_scale
            elif result.match_coverage < 0.45:
                result.score -= 0.10 * penalty_scale
            if result.match_coverage < 0.25 and result.projected_area_ratio is not None:
                result.score -= 0.18 * penalty_scale
            if result.homography_quality < 0.70:
                result.score -= 0.04 * penalty_scale
            if result.evidence_score < 0.25:
                result.score -= 0.18 * penalty_scale
            elif result.evidence_score < 0.45:
                result.score -= 0.08 * penalty_scale
            if result.spatial_distribution_score < 0.20:
                result.score -= 0.16 * penalty_scale
            elif result.spatial_distribution_score < 0.35:
                result.score -= 0.07 * penalty_scale
            if result.localization_status in {"invalid_projection", "invalid_area", "unreliable_homography"}:
                result.score -= 0.20 * penalty_scale
            if float(result.params.get("ratio", 0.75)) > 0.90:
                result.score -= 0.08 * penalty_scale
            if float(result.params.get("ransacReprojThreshold", 5.0)) > 5.0:
                result.score -= 0.04 * penalty_scale
            result.score = min(result.score, self._confidence_cap(result))
            self._assign_confidence(result)

        return sorted(valid_results, key=lambda item: item.score, reverse=True)

    def _weights(self) -> dict[str, float]:
        if self.custom_weights is not None:
            return self.custom_weights

        if self.objective == "时间最快":
            return {
                "inlier_ratio": 0.12,
                "inlier_count": 0.08,
                "match_count": 0.04,
                "reprojection": 0.12,
                "geometry": 0.10,
                "coverage": 0.08,
                "evidence": 0.04,
                "spatial_distribution": 0.02,
                "speed": 0.44,
                "strictness": 0.00,
            }
        if self.objective == "精度最高":
            return {
                "inlier_ratio": 0.24,
                "inlier_count": 0.10,
                "match_count": 0.03,
                "reprojection": 0.24,
                "geometry": 0.19,
                "coverage": 0.16,
                "evidence": 0.08,
                "spatial_distribution": 0.06,
                "speed": 0.02,
                "strictness": 0.00,
            }
        if self.objective == "数量最优":
            return {
                "inlier_ratio": 0.15,
                "inlier_count": 0.30,
                "match_count": 0.25,
                "reprojection": 0.08,
                "geometry": 0.08,
                "coverage": 0.08,
                "evidence": 0.05,
                "spatial_distribution": 0.03,
                "speed": 0.04,
                "strictness": 0.00,
            }
        return {
            "inlier_ratio": 0.26,
            "inlier_count": 0.15,
            "match_count": 0.06,
            "reprojection": 0.17,
            "geometry": 0.16,
            "coverage": 0.14,
            "evidence": 0.07,
            "spatial_distribution": 0.05,
            "speed": 0.04,
            "strictness": 0.00,
        }

    def _penalty_scale(self) -> float:
        if self.objective == "数量最优":
            return 0.65
        if self.objective == "时间最快":
            return 0.75
        return 1.0

    def _reprojection_score(self, reprojection_error: float | None) -> float:
        if reprojection_error is None or not math.isfinite(reprojection_error):
            return 0.0
        return max(0.0, min(1.0, 1.0 / (1.0 + reprojection_error / 3.0)))

    def _coverage_score(self, coverage: float) -> float:
        if coverage <= 0:
            return 0.0
        return max(0.0, min(1.0, coverage / 0.75))

    def _strictness_score(self, result: MatchResult) -> float:
        ratio = float(result.params.get("ratio", 0.75))
        ransac = float(result.params.get("ransacReprojThreshold", 5.0))
        ratio_score = 1.0 - max(0.0, min(1.0, (ratio - 0.70) / 0.25))
        ransac_score = 1.0 - max(0.0, min(1.0, (ransac - 3.0) / 4.0))
        return 0.55 * ratio_score + 0.45 * ransac_score

    def _confidence_cap(self, result: MatchResult) -> float:
        cap = 1.0
        strong_geometry = (
            result.homography_quality >= 0.78
            and result.match_coverage >= 0.50
            and result.projected_inside_ratio >= 0.75
            and result.inlier_ratio >= 0.60
            and result.reprojection_error is not None
            and result.reprojection_error <= 2.0
        )
        if result.num_inliers < 4:
            cap = min(cap, 0.25)
        elif result.num_inliers < 8:
            cap = min(cap, 0.65 if strong_geometry else 0.48)
        elif result.num_inliers < 12:
            cap = min(cap, 0.62)

        if result.num_good_matches < 8:
            cap = min(cap, 0.48)
        elif result.num_good_matches < 12:
            cap = min(cap, 0.68 if strong_geometry else 0.62)

        if result.match_coverage < 0.15:
            cap = min(cap, 0.42)
        elif result.match_coverage < 0.30:
            cap = min(cap, 0.58)

        if result.homography_quality < 0.30:
            cap = min(cap, 0.55)
        if result.projected_inside_ratio < 0.50:
            cap = min(cap, 0.50)
        if result.localization_status in {"invalid_projection", "invalid_area", "unreliable_homography"}:
            cap = min(cap, 0.45)
        if result.evidence_score < 0.25:
            cap = min(cap, 0.45)
        elif result.evidence_score < 0.45:
            cap = min(cap, 0.62)
        if result.spatial_distribution_score < 0.20:
            cap = min(cap, 0.50)
        return cap

    def _assign_confidence(self, result: MatchResult) -> None:
        reasons = []
        if result.num_inliers < 8:
            reasons.append("内点数量偏少")
        if result.evidence_score < 0.45:
            reasons.append("证据充分性不足")
        if result.spatial_distribution_score < 0.35:
            reasons.append("内点分布不够均匀")
        if result.match_coverage < 0.30:
            reasons.append("覆盖度偏低")
        if result.homography_quality < 0.55:
            reasons.append("投影框几何质量一般")
        if result.reprojection_error is not None and result.reprojection_error > 5.0:
            reasons.append("重投影误差偏高")

        if (
            result.score >= 0.70
            and result.evidence_score >= 0.65
            and result.spatial_distribution_score >= 0.50
            and result.homography_quality >= 0.70
            and result.num_inliers >= 16
        ):
            result.confidence_level = "high"
            result.confidence_reason = "证据数量、空间分布和几何质量均较好。"
            return

        if (
            result.score >= 0.52
            and result.evidence_score >= 0.45
            and result.spatial_distribution_score >= 0.30
            and result.homography_quality >= 0.45
            and result.num_inliers >= 8
        ):
            result.confidence_level = "medium"
            result.confidence_reason = "结果可用，但仍需关注局部匹配或框稳定性。"
            return

        if result.score >= 0.35 and result.num_inliers >= 4:
            result.confidence_level = "low"
            result.confidence_reason = "；".join(reasons) if reasons else "结果仅具备弱定位证据。"
            return

        result.confidence_level = "unreliable"
        result.confidence_reason = "；".join(reasons) if reasons else "综合证据不足。"

    def _normalize_weights(
        self,
        weights: dict[str, float] | None,
    ) -> dict[str, float] | None:
        if not weights:
            return None

        allowed_names = {
            "inlier_ratio",
            "inlier_count",
            "match_count",
            "reprojection",
            "geometry",
            "coverage",
            "evidence",
            "spatial_distribution",
            "speed",
            "strictness",
        }
        sanitized = {}
        for name in allowed_names:
            value = weights.get(name)
            if value is None:
                value = 0.0
            sanitized[name] = max(float(value), 0.0)

        total = sum(sanitized.values())
        if total <= 0:
            return None
        return {name: value / total for name, value in sanitized.items()}
