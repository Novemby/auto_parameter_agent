from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import cv2

from agents.evaluator import EvaluatorAgent
from agents.llm_strategy import LLMStrategyAgent, StrategyPlan
from agents.types import AlgorithmCandidate, ImageProfile, MatchResult
from algorithms.feature_matcher import FeatureMatcher


class TuningAgent:
    """Runs coarse-to-fine candidate search with early stopping."""

    def __init__(
        self,
        max_combinations: int | None = None,
        objective: str = "综合平衡",
        strategy_agent: LLMStrategyAgent | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.max_combinations = max_combinations
        self.matcher = FeatureMatcher()
        self.evaluator = EvaluatorAgent(objective=objective)
        self.strategy_agent = strategy_agent
        self.last_strategy_plan: StrategyPlan | None = None
        self.progress_callback = progress_callback

    def run(
        self,
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        candidates: list[AlgorithmCandidate],
        image_profile: ImageProfile | None = None,
        object_mask: cv2.Mat | None = None,
    ) -> list[MatchResult]:
        budget = self.max_combinations
        coarse_limit = None if budget is None else max(6, budget // 2)
        coarse_candidates = self._coarse_candidates(candidates, coarse_limit)
        seen = {self._candidate_key(candidate) for candidate in coarse_candidates}
        self._emit(
            agent="TuningAgent",
            action="开始粗搜",
            phase="coarse",
            evaluated_count=0,
            total_count=len(coarse_candidates),
        )
        results = self._run_candidates(
            candidates=coarse_candidates,
            object_image=object_image,
            scene_image=scene_image,
            object_mask=object_mask,
            phase="coarse",
            start_step=1,
        )
        ranked = self.evaluator.score(results)
        self._emit_best(ranked, "完成粗搜", "coarse", len(results), len(coarse_candidates))

        remaining_budget = None if budget is None else max(budget - len(results), 0)
        if remaining_budget == 0 or not ranked:
            return ranked

        if self.strategy_agent is not None:
            self._emit(
                agent="LLMStrategyAgent",
                action="根据粗搜结果生成下一轮参数建议",
                phase="strategy",
                evaluated_count=len(results),
            )
            failed_results = [result for result in results if result.error is not None]
            self.last_strategy_plan = self.strategy_agent.suggest(
                image_profile=image_profile,
                ranked_results=ranked[:5],
                failed_results=failed_results,
                seen_keys=seen,
            )
            strategy_candidates = [
                AlgorithmCandidate(
                    name=suggestion.algorithm,
                    params=suggestion.params,
                )
                for suggestion in self.last_strategy_plan.suggestions
            ]
            if remaining_budget is not None:
                strategy_candidates = strategy_candidates[:remaining_budget]
            if strategy_candidates:
                total = len(results) + len(strategy_candidates)
                strategy_results = self._run_candidates(
                    candidates=strategy_candidates,
                    object_image=object_image,
                    scene_image=scene_image,
                    object_mask=object_mask,
                    phase=(
                        "llm-strategy"
                        if self.last_strategy_plan.used_llm
                        else "strategy"
                    ),
                    start_step=len(results) + 1,
                )
                results.extend(strategy_results)
                ranked = self.evaluator.score(results)
                self._emit_best(
                    ranked,
                    "完成策略候选评估",
                    "llm-strategy" if self.last_strategy_plan.used_llm else "strategy",
                    len(results),
                    total,
                )
                if remaining_budget is not None:
                    remaining_budget -= len(strategy_results)
                    if remaining_budget <= 0:
                        return ranked

        best_score = ranked[0].score
        no_improvement_rounds = 0
        step = len(results) + 1

        for round_index in range(1, 4):
            fine_candidates = self._fine_candidates(ranked[:3], seen, round_index)
            if remaining_budget is not None:
                fine_candidates = fine_candidates[:remaining_budget]
            if not fine_candidates:
                break

            self._emit(
                agent="TuningAgent",
                action=f"开始第 {round_index} 轮细搜",
                phase=f"fine-{round_index}",
                evaluated_count=len(results),
                total_count=(
                    len(results) + len(fine_candidates)
                    if remaining_budget is not None
                    else None
                ),
            )
            fine_results = self._run_candidates(
                candidates=fine_candidates,
                object_image=object_image,
                scene_image=scene_image,
                object_mask=object_mask,
                phase=f"fine-{round_index}",
                start_step=step,
            )
            step += len(fine_results)
            results.extend(fine_results)
            ranked = self.evaluator.score(results)
            self._emit_best(
                ranked,
                f"完成第 {round_index} 轮细搜",
                f"fine-{round_index}",
                len(results),
                (
                    len(results)
                    if remaining_budget is not None
                    else None
                ),
            )

            improvement = ranked[0].score - best_score
            if improvement < 0.005:
                no_improvement_rounds += 1
            else:
                no_improvement_rounds = 0
                best_score = ranked[0].score

            if remaining_budget is not None:
                remaining_budget -= len(fine_results)
                if remaining_budget <= 0:
                    break
            if no_improvement_rounds >= 2:
                break

        return ranked

    def _run_candidates(
        self,
        candidates: list[AlgorithmCandidate],
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        object_mask: cv2.Mat | None,
        phase: str,
        start_step: int,
    ) -> list[MatchResult]:
        results = []
        for offset, candidate in enumerate(candidates):
            current_step = start_step + offset
            self._emit(
                agent="TuningAgent",
                action=f"运行 {candidate.name} 参数组合",
                phase=phase,
                search_step=current_step,
                evaluated_count=current_step - 1,
                total_count=start_step + len(candidates) - 1,
                algorithm=candidate.name,
                params=candidate.params,
            )
            result = self.matcher.run(
                algorithm=candidate.name,
                params=candidate.params,
                object_image=object_image,
                scene_image=scene_image,
                object_mask=object_mask,
            )
            result.search_phase = phase
            result.search_step = current_step
            results.append(result)
        return results

    def _emit(self, **event: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

    def _emit_best(
        self,
        ranked: list[MatchResult],
        action: str,
        phase: str,
        evaluated_count: int,
        total_count: int | None,
    ) -> None:
        best = ranked[0] if ranked else None
        self._emit(
            agent="EvaluatorAgent",
            action=action,
            phase=phase,
            evaluated_count=evaluated_count,
            total_count=total_count,
            best_algorithm=best.algorithm if best is not None else None,
            best_score=round(best.score, 4) if best is not None else None,
        )

    def _coarse_candidates(
        self,
        candidates: list[AlgorithmCandidate],
        budget: int | None,
    ) -> list[AlgorithmCandidate]:
        if budget is None or budget >= len(candidates):
            return candidates

        by_algorithm: dict[str, list[AlgorithmCandidate]] = {}
        for candidate in candidates:
            by_algorithm.setdefault(candidate.name, []).append(candidate)

        selected = []
        per_algorithm = max(1, budget // max(len(by_algorithm), 1))
        for algorithm_candidates in by_algorithm.values():
            if len(algorithm_candidates) <= per_algorithm:
                selected.extend(algorithm_candidates)
                continue
            stride = max(len(algorithm_candidates) - 1, 1) / max(per_algorithm - 1, 1)
            indexes = sorted({round(index * stride) for index in range(per_algorithm)})
            selected.extend(algorithm_candidates[index] for index in indexes)

        return selected[:budget]

    def _fine_candidates(
        self,
        best_results: list[MatchResult],
        seen: set[str],
        round_index: int,
    ) -> list[AlgorithmCandidate]:
        candidates = []
        for result in best_results:
            for params in self._neighbor_params(result.algorithm, result.params, round_index):
                candidate = AlgorithmCandidate(name=result.algorithm, params=params)
                key = self._candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _neighbor_params(
        self,
        algorithm: str,
        params: dict[str, Any],
        round_index: int,
    ) -> list[dict[str, Any]]:
        if algorithm == "SUPERPOINT_SUPERGLUE":
            return self._superpoint_superglue_neighbors(params, round_index)

        ratio_step = 0.03 if round_index == 1 else 0.015
        ransac_step = 0.75 if round_index == 1 else 0.35
        base_ratio = float(params.get("ratio", 0.75))
        base_ransac = float(params.get("ransacReprojThreshold", 5.0))
        max_ratio = 0.90 if algorithm in {"AKAZE", "ORB"} else 0.88
        ratios = self._clamped_values(base_ratio, ratio_step, 0.55, max_ratio)
        ransac_values = self._clamped_values(base_ransac, ransac_step, 2.0, 6.0)

        variants = []
        for ratio in ratios:
            for ransac in ransac_values:
                new_params = dict(params)
                new_params["ratio"] = ratio
                new_params["ransacReprojThreshold"] = ransac
                variants.append(new_params)

        if algorithm == "SIFT":
            contrast = float(params.get("contrastThreshold", 0.04))
            for contrast_value in self._clamped_values(contrast, contrast * 0.35, 0.005, 0.12):
                new_params = dict(params)
                new_params["contrastThreshold"] = contrast_value
                variants.append(new_params)
        elif algorithm == "SURF":
            hessian = float(params.get("hessianThreshold", 400))
            for hessian_value in self._clamped_values(hessian, hessian * 0.35, 80, 1200):
                new_params = dict(params)
                new_params["hessianThreshold"] = int(hessian_value)
                variants.append(new_params)
        elif algorithm == "ORB":
            scale = float(params.get("scaleFactor", 1.2))
            for scale_value in self._clamped_values(scale, 0.08, 1.05, 1.6):
                new_params = dict(params)
                new_params["scaleFactor"] = scale_value
                variants.append(new_params)
        elif algorithm == "FAST_BRISK":
            threshold = float(params.get("fastThreshold", 20))
            for threshold_value in self._clamped_values(threshold, 8, 5, 80):
                new_params = dict(params)
                new_params["fastThreshold"] = int(threshold_value)
                variants.append(new_params)
        elif algorithm == "AKAZE":
            threshold = float(params.get("threshold", 0.001))
            for threshold_value in self._clamped_values(threshold, threshold * 0.5, 0.0001, 0.006):
                new_params = dict(params)
                new_params["threshold"] = threshold_value
                variants.append(new_params)

        return variants

    def _clamped_values(
        self,
        center: float,
        step: float,
        minimum: float,
        maximum: float,
    ) -> list[float]:
        values = []
        for value in [center - step, center, center + step]:
            value = min(max(value, minimum), maximum)
            values.append(round(value, 6))
        return sorted(set(values))

    def _candidate_key(self, candidate: AlgorithmCandidate) -> str:
        return f"{candidate.name}:{json.dumps(candidate.params, sort_keys=True)}"

    def _superpoint_superglue_neighbors(
        self,
        params: dict[str, Any],
        round_index: int,
    ) -> list[dict[str, Any]]:
        keypoint = float(params.get("keypointThreshold", 0.005))
        match = float(params.get("matchThreshold", 0.2))
        ransac = float(params.get("ransacReprojThreshold", 5.0))
        keypoint_step = 0.002 if round_index == 1 else 0.001
        match_step = 0.08 if round_index == 1 else 0.04
        ransac_step = 0.75 if round_index == 1 else 0.35

        variants = []
        for keypoint_value in self._clamped_values(keypoint, keypoint_step, 0.001, 0.02):
            for match_value in self._clamped_values(match, match_step, 0.1, 0.6):
                for ransac_value in self._clamped_values(ransac, ransac_step, 2.0, 6.0):
                    new_params = dict(params)
                    new_params["keypointThreshold"] = keypoint_value
                    new_params["matchThreshold"] = match_value
                    new_params["ransacReprojThreshold"] = ransac_value
                    variants.append(new_params)
        return variants
