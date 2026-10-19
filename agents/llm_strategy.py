from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agents.settings import get_deepseek_api_key
from agents.types import AlgorithmCandidate, ImageProfile, MatchResult


@dataclass(frozen=True)
class StrategySuggestion:
    algorithm: str
    params: dict[str, Any]
    reason: str
    source: str
    base_score: float | None = None


@dataclass
class StrategyPlan:
    suggestions: list[StrategySuggestion] = field(default_factory=list)
    reason: str = ""
    used_llm: bool = False
    error: str | None = None


class LLMStrategyAgent:
    """Suggests the next tuning candidates from scored matching results."""

    SUPPORTED_ALGORITHMS = {
        "SIFT",
        "SURF",
        "ORB",
        "FAST_BRISK",
        "AKAZE",
        "SUPERPOINT_SUPERGLUE",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: int = 45,
        max_suggestions: int = 6,
    ) -> None:
        self.api_key = api_key or get_deepseek_api_key()
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_suggestions = max_suggestions

    def suggest(
        self,
        image_profile: ImageProfile | None,
        ranked_results: list[MatchResult],
        failed_results: list[MatchResult],
        seen_keys: set[str],
    ) -> StrategyPlan:
        if not ranked_results:
            return StrategyPlan(
                suggestions=[],
                reason="No valid ranked result is available for strategy tuning.",
                used_llm=False,
            )

        if self.api_key:
            try:
                plan = self._call_llm(
                    image_profile=image_profile,
                    ranked_results=ranked_results,
                    failed_results=failed_results,
                    seen_keys=seen_keys,
                )
                if plan.suggestions:
                    return plan
            except (
                urllib.error.URLError,
                TimeoutError,
                RuntimeError,
                KeyError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                fallback = self._fallback_strategy(ranked_results, seen_keys)
                fallback.error = f"LLM strategy failed: {exc}"
                return fallback

        return self._fallback_strategy(ranked_results, seen_keys)

    def _call_llm(
        self,
        image_profile: ImageProfile | None,
        ranked_results: list[MatchResult],
        failed_results: list[MatchResult],
        seen_keys: set[str],
    ) -> StrategyPlan:
        context = self._build_context(image_profile, ranked_results, failed_results)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an automatic parameter tuning strategy agent for "
                        "feature matching. Use the metrics to suggest the next small "
                        "parameter changes. Return strict JSON only. Do not invent "
                        "unsupported algorithms or parameters."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Given the JSON experiment state, suggest up to "
                        f"{self.max_suggestions} next tuning candidates. "
                        "Return this exact JSON shape: "
                        '{"recommendations":[{"base_rank":1,"algorithm":"SIFT",'
                        '"param_updates":{"ratio":0.72},"reason":"..."}]}. '
                        "Prefer small changes around the top results. "
                        "If good matches are many but inlier ratio is low, tighten ratio "
                        "or RANSAC. If matches are too few, increase features or relax ratio. "
                        "If reprojection error is high, reduce RANSAC threshold. "
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
        suggestions = self._suggestions_from_payload(
            payload=parsed,
            ranked_results=ranked_results,
            seen_keys=seen_keys,
            source="llm",
        )
        return StrategyPlan(
            suggestions=suggestions,
            reason=f"LLM strategy generated {len(suggestions)} validated candidate(s).",
            used_llm=True,
        )

    def _fallback_strategy(
        self,
        ranked_results: list[MatchResult],
        seen_keys: set[str],
    ) -> StrategyPlan:
        payload: dict[str, Any] = {"recommendations": []}
        for rank, result in enumerate(ranked_results[:3], start=1):
            for updates, reason in self._heuristic_updates(result):
                payload["recommendations"].append(
                    {
                        "base_rank": rank,
                        "algorithm": result.algorithm,
                        "param_updates": updates,
                        "reason": reason,
                    }
                )
        suggestions = self._suggestions_from_payload(
            payload=payload,
            ranked_results=ranked_results,
            seen_keys=seen_keys,
            source="local_fallback",
        )
        return StrategyPlan(
            suggestions=suggestions[: self.max_suggestions],
            reason=(
                "No API key was configured, so a local strategy heuristic generated "
                f"{len(suggestions[: self.max_suggestions])} candidate(s)."
            ),
            used_llm=False,
        )

    def _heuristic_updates(self, result: MatchResult) -> list[tuple[dict[str, Any], str]]:
        params = result.params
        updates: list[tuple[dict[str, Any], str]] = []
        ratio = float(params.get("ratio", 0.75))
        ransac = float(params.get("ransacReprojThreshold", 5.0))

        if result.num_good_matches >= 20 and result.inlier_ratio < 0.35:
            updates.append(
                (
                    {
                        "ratio": round(max(ratio - 0.05, 0.55), 6),
                        "ransacReprojThreshold": round(max(ransac - 0.5, 2.0), 6),
                    },
                    "Many tentative matches but a low inlier ratio; tighten match filtering.",
                )
            )
        if result.num_good_matches < 12:
            updates.append(
                (
                    {
                        "ratio": round(min(ratio + 0.04, 0.9), 6),
                        "ransacReprojThreshold": round(min(ransac + 0.35, 6.0), 6),
                    },
                    "Too few good matches; slightly relax filtering to collect more evidence.",
                )
            )
        if result.reprojection_error is not None and result.reprojection_error > 4.0:
            updates.append(
                (
                    {"ransacReprojThreshold": round(max(ransac - 0.75, 2.0), 6)},
                    "High reprojection error; reduce the RANSAC threshold.",
                )
            )
        if result.match_coverage < 0.30:
            updates.extend(self._coverage_updates(result))
        if not updates:
            updates.append(
                (
                    {
                        "ratio": round(max(ratio - 0.025, 0.55), 6),
                        "ransacReprojThreshold": round(max(ransac - 0.25, 2.0), 6),
                    },
                    "Top result is acceptable; probe a stricter nearby candidate.",
                )
            )
        return updates

    def _coverage_updates(self, result: MatchResult) -> list[tuple[dict[str, Any], str]]:
        params = result.params
        if result.algorithm in {"SIFT", "ORB"}:
            return [
                (
                    {"nfeatures": int(params.get("nfeatures", 1000)) + 500},
                    "Low spatial coverage; increase the feature budget.",
                )
            ]
        if result.algorithm == "FAST_BRISK":
            return [
                (
                    {"fastThreshold": int(max(int(params.get("fastThreshold", 20)) - 6, 5))},
                    "Low spatial coverage; lower FAST threshold to detect more corners.",
                )
            ]
        if result.algorithm == "AKAZE":
            return [
                (
                    {
                        "threshold": round(
                            max(float(params.get("threshold", 0.001)) * 0.6, 0.0001),
                            6,
                        )
                    },
                    "Low spatial coverage; lower AKAZE threshold to retain more features.",
                )
            ]
        if result.algorithm == "SURF":
            return [
                (
                    {
                        "hessianThreshold": int(
                            max(int(params.get("hessianThreshold", 400)) * 0.75, 80)
                        )
                    },
                    "Low spatial coverage; lower SURF hessian threshold.",
                )
            ]
        if result.algorithm == "SUPERPOINT_SUPERGLUE":
            return [
                (
                    {
                        "keypointThreshold": round(
                            max(float(params.get("keypointThreshold", 0.005)) - 0.002, 0.001),
                            6,
                        )
                    },
                    "Low spatial coverage; lower SuperPoint keypoint threshold.",
                )
            ]
        return []

    def _suggestions_from_payload(
        self,
        payload: dict[str, Any],
        ranked_results: list[MatchResult],
        seen_keys: set[str],
        source: str,
    ) -> list[StrategySuggestion]:
        suggestions = []
        recommendations = payload.get("recommendations", [])
        if not isinstance(recommendations, list):
            return []

        for item in recommendations:
            if not isinstance(item, dict):
                continue
            base = self._select_base_result(item, ranked_results)
            if base is None:
                continue
            algorithm = str(item.get("algorithm") or base.algorithm)
            if algorithm != base.algorithm or algorithm not in self.SUPPORTED_ALGORITHMS:
                continue
            updates = item.get("param_updates") or item.get("updates") or {}
            if not isinstance(updates, dict):
                continue

            params = dict(base.params)
            for name, value in updates.items():
                sanitized = self._sanitize_param(algorithm, name, value)
                if sanitized is not None:
                    params[name] = sanitized

            candidate = AlgorithmCandidate(name=algorithm, params=params)
            key = self._candidate_key(candidate)
            if key in seen_keys or params == base.params:
                continue
            seen_keys.add(key)
            suggestions.append(
                StrategySuggestion(
                    algorithm=algorithm,
                    params=params,
                    reason=str(item.get("reason") or "Strategy candidate."),
                    source=source,
                    base_score=base.score,
                )
            )
            if len(suggestions) >= self.max_suggestions:
                break
        return suggestions

    def _select_base_result(
        self,
        item: dict[str, Any],
        ranked_results: list[MatchResult],
    ) -> MatchResult | None:
        try:
            rank = int(item.get("base_rank", 1))
        except (TypeError, ValueError):
            rank = 1
        if rank < 1 or rank > len(ranked_results):
            return None
        return ranked_results[rank - 1]

    def _sanitize_param(self, algorithm: str, name: str, value: Any) -> Any | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            if name in {"matcher", "weights"} and isinstance(value, str):
                if name == "matcher" and value in {"BF", "FLANN"}:
                    return value
                if name == "weights" and value in {"indoor", "outdoor"}:
                    return value
            return None

        if name == "ratio":
            maximum = 0.90 if algorithm in {"AKAZE", "ORB"} else 0.88
            return round(self._clamp(numeric, 0.55, maximum), 6)
        if name == "ransacReprojThreshold":
            return round(self._clamp(numeric, 2.0, 6.0), 6)
        if name == "nfeatures":
            return int(self._clamp(round(numeric), 200, 3000))
        if algorithm == "SIFT" and name == "contrastThreshold":
            return round(self._clamp(numeric, 0.005, 0.12), 6)
        if algorithm == "SURF" and name == "hessianThreshold":
            return int(self._clamp(round(numeric), 80, 1200))
        if algorithm == "ORB" and name == "scaleFactor":
            return round(self._clamp(numeric, 1.05, 1.6), 6)
        if algorithm == "FAST_BRISK" and name == "fastThreshold":
            return int(self._clamp(round(numeric), 5, 80))
        if algorithm == "AKAZE" and name == "threshold":
            return round(self._clamp(numeric, 0.0001, 0.006), 6)
        if algorithm == "SUPERPOINT_SUPERGLUE" and name == "keypointThreshold":
            return round(self._clamp(numeric, 0.001, 0.02), 6)
        if algorithm == "SUPERPOINT_SUPERGLUE" and name == "matchThreshold":
            return round(self._clamp(numeric, 0.1, 0.6), 6)
        return None

    def _build_context(
        self,
        image_profile: ImageProfile | None,
        ranked_results: list[MatchResult],
        failed_results: list[MatchResult],
    ) -> dict[str, Any]:
        profile = None
        if image_profile is not None:
            profile = {
                "object_size": image_profile.object_size,
                "scene_size": image_profile.scene_size,
                "object_edge_density": round(image_profile.object_edge_density, 6),
                "scene_edge_density": round(image_profile.scene_edge_density, 6),
                "object_texture_score": round(image_profile.object_texture_score, 3),
                "scene_texture_score": round(image_profile.scene_texture_score, 3),
            }
        return {
            "image_profile": profile,
            "top_results": [
                self._result_brief(rank, result)
                for rank, result in enumerate(ranked_results[:5], start=1)
            ],
            "failed_results": [
                {
                    "algorithm": result.algorithm,
                    "phase": result.search_phase,
                    "error": result.error,
                    "params": result.params,
                }
                for result in failed_results[:6]
            ],
        }

    def _result_brief(self, rank: int, result: MatchResult) -> dict[str, Any]:
        return {
            "rank": rank,
            "algorithm": result.algorithm,
            "phase": result.search_phase,
            "score": round(result.score, 6),
            "params": result.params,
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
            "confidence_level": result.confidence_level,
            "runtime_ms": round(result.runtime_ms, 3),
        }

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

    def _candidate_key(self, candidate: AlgorithmCandidate) -> str:
        return f"{candidate.name}:{json.dumps(candidate.params, sort_keys=True)}"

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return min(max(value, minimum), maximum)
