from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from agents.settings import get_deepseek_api_key
from agents.types import ImageProfile, MatchResult


WEIGHT_NAMES = {
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


@dataclass(frozen=True)
class ScoringPolicy:
    weights: dict[str, float]
    reason: str
    used_llm: bool = False
    error: str | None = None


class ScoringPolicyAgent:
    """Generates adaptive scoring weights from image and coarse-result signals."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: int = 45,
        use_llm: bool = True,
    ) -> None:
        self.api_key = api_key or get_deepseek_api_key()
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.use_llm = use_llm

    def decide(
        self,
        image_profile: ImageProfile,
        objective: str,
        ranked_results: list[MatchResult],
    ) -> ScoringPolicy:
        if self.use_llm and self.api_key:
            try:
                policy = self._call_llm(image_profile, objective, ranked_results)
                if policy is not None:
                    return policy
            except (
                urllib.error.URLError,
                TimeoutError,
                RuntimeError,
                KeyError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                fallback = self._fallback_policy(image_profile, objective, ranked_results)
                return ScoringPolicy(
                    weights=fallback.weights,
                    reason=f"{fallback.reason} LLM scoring policy failed: {exc}",
                    used_llm=False,
                    error=str(exc),
                )

        return self._fallback_policy(image_profile, objective, ranked_results)

    def _call_llm(
        self,
        image_profile: ImageProfile,
        objective: str,
        ranked_results: list[MatchResult],
    ) -> ScoringPolicy | None:
        context = self._build_context(image_profile, objective, ranked_results)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a scoring-policy agent for feature matching. "
                        "Choose deterministic metric weights for the evaluator. "
                        "Return strict JSON only. Do not rank final results directly."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Generate scoring weights for these exact metrics: "
                        "inlier_ratio, inlier_count, match_count, reprojection, "
                        "geometry, coverage, evidence, spatial_distribution, "
                        "speed, strictness. "
                        "Weights must be non-negative and sum to 1. "
                        "Return JSON shape: "
                        '{"weights":{"inlier_ratio":0.2,"inlier_count":0.1,'
                        '"match_count":0.05,"reprojection":0.2,"geometry":0.2,'
                        '"coverage":0.12,"evidence":0.05,'
                        '"spatial_distribution":0.03,"speed":0.03,'
                        '"strictness":0.02},'
                        '"reason":"..."}.\n'
                        f"Experiment state:\n{json.dumps(context, ensure_ascii=False)}"
                    ),
                },
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        content = data["choices"][0]["message"]["content"].strip()
        parsed = self._parse_json_object(content)
        weights = self._sanitize_weights(parsed.get("weights", {}))
        if weights is None:
            return None
        return ScoringPolicy(
            weights=weights,
            reason=str(parsed.get("reason") or "LLM generated adaptive scoring weights."),
            used_llm=True,
        )

    def _fallback_policy(
        self,
        image_profile: ImageProfile,
        objective: str,
        ranked_results: list[MatchResult],
    ) -> ScoringPolicy:
        if objective == "时间最快":
            weights = {
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
            return ScoringPolicy(weights=weights, reason="速度优先目标，提高耗时指标权重。")

        texture = min(image_profile.object_texture_score, image_profile.scene_texture_score)
        edge_density = min(image_profile.object_edge_density, image_profile.scene_edge_density)
        best = ranked_results[0] if ranked_results else None

        weights = {
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
        reasons = ["使用本地规则生成自适应评分权重。"]

        if objective == "精度最高":
            weights.update(
                {
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
            )
            reasons.append("精度优先目标，提高重投影误差、几何质量和覆盖度权重。")
        elif objective == "数量最优":
            weights.update(
                {
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
            )
            reasons.append("数量优先目标，提高优质匹配数和内点数权重。")

        if texture < 120 or edge_density < 0.025:
            weights["match_count"] *= 0.65
            weights["inlier_count"] *= 0.80
            weights["geometry"] *= 1.25
            weights["coverage"] *= 1.25
            weights["reprojection"] *= 1.15
            weights["evidence"] *= 1.10
            reasons.append("图像纹理或边缘较弱，降低数量指标，强调几何合理性和覆盖度。")

        if best is not None and best.match_coverage < 0.30:
            weights["coverage"] *= 1.35
            weights["geometry"] *= 1.15
            weights["spatial_distribution"] *= 1.25
            weights["match_count"] *= 0.85
            reasons.append("初步结果覆盖度偏低，提高覆盖度和框合理性权重。")

        if best is not None and best.inlier_ratio < 0.35:
            weights["inlier_ratio"] *= 1.25
            weights["strictness"] *= 1.30
            weights["match_count"] *= 0.75
            reasons.append("初步结果内点率偏低，提高内点率和参数严格性权重。")

        sanitized = self._sanitize_weights(weights)
        return ScoringPolicy(
            weights=sanitized or weights,
            reason=" ".join(reasons),
            used_llm=False,
        )

    def _build_context(
        self,
        image_profile: ImageProfile,
        objective: str,
        ranked_results: list[MatchResult],
    ) -> dict[str, Any]:
        return {
            "objective": objective,
            "image_profile": {
                "object_size": image_profile.object_size,
                "scene_size": image_profile.scene_size,
                "object_edge_density": round(image_profile.object_edge_density, 6),
                "scene_edge_density": round(image_profile.scene_edge_density, 6),
                "object_texture_score": round(image_profile.object_texture_score, 3),
                "scene_texture_score": round(image_profile.scene_texture_score, 3),
            },
            "top_results": [
                {
                    "rank": rank,
                    "algorithm": result.algorithm,
                    "score": round(result.score, 6),
                    "good_matches": result.num_good_matches,
                    "inliers": result.num_inliers,
                    "inlier_ratio": round(result.inlier_ratio, 6),
                    "reprojection_error": None
                    if result.reprojection_error is None
                    else round(result.reprojection_error, 6),
            "match_coverage": round(result.match_coverage, 6),
            "homography_quality": round(result.homography_quality, 6),
            "evidence_score": round(result.evidence_score, 6),
            "spatial_distribution_score": round(result.spatial_distribution_score, 6),
            "runtime_ms": round(result.runtime_ms, 3),
                }
                for rank, result in enumerate(ranked_results[:5], start=1)
            ],
        }

    def _sanitize_weights(self, weights: Any) -> dict[str, float] | None:
        if not isinstance(weights, dict):
            return None
        sanitized = {}
        for name in WEIGHT_NAMES:
            try:
                value = float(weights.get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            sanitized[name] = max(value, 0.0)
        total = sum(sanitized.values())
        if total <= 0:
            return None
        return {name: value / total for name, value in sanitized.items()}

    def _parse_json_object(self, content: str) -> dict[str, Any]:
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM response did not contain a JSON object.")
        return json.loads(content[start : end + 1])
