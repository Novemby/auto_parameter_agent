from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agents.algorithm_selector import AlgorithmSelectorAgent
from agents.decision_log import DecisionLog
from agents.deep_policy import DeepPolicyAgent
from agents.evaluator import EvaluatorAgent
from agents.image_analyzer import ImageAnalyzerAgent
from agents.llm_strategy import LLMStrategyAgent, StrategyPlan
from agents.reporter import ReporterAgent
from agents.scoring_policy import ScoringPolicy, ScoringPolicyAgent
from agents.tuner import TuningAgent
from agents.types import ImageProfile, MatchResult


@dataclass
class PipelineOutput:
    image_profile: ImageProfile
    results: list[MatchResult]
    failed_results: list[MatchResult]
    best_result: MatchResult | None
    objective: str
    deep_learning_used: bool
    deep_learning_reason: str
    strategy_plan: StrategyPlan | None
    scoring_policy: ScoringPolicy | None
    decision_log: DecisionLog
    object_image_used: cv2.Mat
    object_crop_box: tuple[int, int, int, int] | None
    object_mask_used: cv2.Mat | None
    object_mask_box: tuple[int, int, int, int] | None
    object_mask_area_ratio: float | None


class AutoParameterPipeline:
    def __init__(
        self,
        max_combinations: int | None = None,
        include_deep_learning: bool = True,
        deep_learning_policy: str = "按需启用",
        objective: str = "综合平衡",
        enable_object_crop: bool = False,
        enable_object_mask: bool = False,
        enable_llm_strategy: bool = False,
        llm_strategy_api_key: str | None = None,
        llm_strategy_model: str = "deepseek-chat",
        enable_llm_scoring_policy: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.max_combinations = max_combinations
        self.include_deep_learning = include_deep_learning
        self.deep_learning_policy = deep_learning_policy
        self.objective = objective
        self.enable_object_crop = enable_object_crop
        self.enable_object_mask = enable_object_mask
        self.enable_llm_strategy = enable_llm_strategy
        self.llm_strategy_api_key = llm_strategy_api_key
        self.llm_strategy_model = llm_strategy_model
        self.enable_llm_scoring_policy = enable_llm_scoring_policy
        self.progress_callback = progress_callback

    def run(self, object_image: cv2.Mat, scene_image: cv2.Mat) -> PipelineOutput:
        decision_log = DecisionLog()
        self._emit(agent="PreprocessAgent", action="准备目标图预处理")
        if self.enable_object_crop:
            object_image_used, crop_box = self._auto_crop_object_image(object_image)
            preprocess_reason = "用户启用目标图自动裁剪，尝试裁掉背景以改善投影框边界。"
        else:
            object_image_used, crop_box = object_image, None
            preprocess_reason = "目标图自动裁剪已关闭，使用上传的原始目标图，避免误裁主体。"
        decision_log.add(
            agent="PreprocessAgent",
            action="目标图预处理",
            reason=preprocess_reason,
            details={
                "crop_box": crop_box,
                "used_cropped_image": crop_box is not None,
                "enable_object_crop": self.enable_object_crop,
            },
        )
        self._emit(
            agent="PreprocessAgent",
            action="完成目标图预处理",
            used_cropped_image=crop_box is not None,
            crop_box=crop_box,
        )
        self._emit(agent="PreprocessAgent", action="生成目标区域 mask")
        object_mask, mask_box = self._auto_mask_object_image(object_image_used)
        if not self.enable_object_mask:
            object_mask, mask_box = None, None
        mask_area_ratio = (
            self._mask_area_ratio(object_mask) if object_mask is not None else None
        )
        decision_log.add(
            agent="PreprocessAgent",
            action="目标区域 mask 生成",
            reason=(
                "启用轻量目标 mask，基于边界背景颜色估计目标区域，用于减少背景特征干扰。"
                if object_mask is not None
                else "目标 mask 未启用或自动生成质量不足，继续使用整张目标图提取特征。"
            ),
            details={
                "enable_object_mask": self.enable_object_mask,
                "mask_box": mask_box,
                "mask_area_ratio": mask_area_ratio,
            },
        )
        self._emit(
            agent="PreprocessAgent",
            action="完成目标区域 mask 生成",
            mask_enabled=object_mask is not None,
            mask_box=mask_box,
            mask_area_ratio=mask_area_ratio,
        )
        self._emit(agent="ImageAnalyzerAgent", action="分析图像尺寸、边缘密度和纹理强度")
        image_profile = ImageAnalyzerAgent().analyze(object_image_used, scene_image)
        decision_log.add(
            agent="ImageAnalyzerAgent",
            action="分析图像特征",
            reason="边缘密度、纹理强度和尺寸用于生成候选算法与参数。",
            details={
                "object_size": image_profile.object_size,
                "scene_size": image_profile.scene_size,
                "object_edge_density": round(image_profile.object_edge_density, 4),
                "scene_edge_density": round(image_profile.scene_edge_density, 4),
                "object_texture_score": round(image_profile.object_texture_score, 2),
                "scene_texture_score": round(image_profile.scene_texture_score, 2),
            },
        )
        self._emit(
            agent="ImageAnalyzerAgent",
            action="完成图像特征分析",
            object_edge_density=round(image_profile.object_edge_density, 4),
            scene_edge_density=round(image_profile.scene_edge_density, 4),
            object_texture_score=round(image_profile.object_texture_score, 2),
            scene_texture_score=round(image_profile.scene_texture_score, 2),
        )
        self._emit(agent="AlgorithmSelectorAgent", action="生成传统算法候选参数")
        selector = AlgorithmSelectorAgent()
        traditional_candidates = selector.select(
            image_profile,
            include_deep_learning=False,
        )
        decision_log.add(
            agent="AlgorithmSelectorAgent",
            action="生成传统算法候选",
            reason="先用轻量传统算法进行粗搜和细搜，降低整体运行成本。",
            details={
                "candidate_count": len(traditional_candidates),
                "algorithms": sorted({candidate.name for candidate in traditional_candidates}),
            },
        )
        self._emit(
            agent="AlgorithmSelectorAgent",
            action="完成传统算法候选生成",
            candidate_count=len(traditional_candidates),
            algorithms=sorted({candidate.name for candidate in traditional_candidates}),
        )
        strategy_agent = None
        if self.enable_llm_strategy:
            strategy_agent = LLMStrategyAgent(
                api_key=self.llm_strategy_api_key,
                model=self.llm_strategy_model,
            )
        tuner = TuningAgent(
            max_combinations=self.max_combinations,
            objective=self.objective,
            strategy_agent=strategy_agent,
            progress_callback=self.progress_callback,
        )
        raw_results = tuner.run(
            object_image=object_image_used,
            scene_image=scene_image,
            candidates=traditional_candidates,
            image_profile=image_profile,
            object_mask=object_mask,
        )
        strategy_plan = tuner.last_strategy_plan
        results = EvaluatorAgent(objective=self.objective).score(raw_results)
        self._emit(
            agent="ScoringPolicyAgent",
            action="根据图像特征和初步结果生成评分权重",
            evaluated_count=len(raw_results),
            best_algorithm=results[0].algorithm if results else None,
            best_score=round(results[0].score, 4) if results else None,
        )
        scoring_policy = ScoringPolicyAgent(
            api_key=self.llm_strategy_api_key,
            model=self.llm_strategy_model,
            use_llm=self.enable_llm_scoring_policy,
        ).decide(
            image_profile=image_profile,
            objective=self.objective,
            ranked_results=results,
        )
        results = EvaluatorAgent(
            objective=self.objective,
            custom_weights=scoring_policy.weights,
        ).score(raw_results)
        self._emit(
            agent="EvaluatorAgent",
            action="使用自适应评分权重重新排序",
            evaluated_count=len(raw_results),
            best_algorithm=results[0].algorithm if results else None,
            best_score=round(results[0].score, 4) if results else None,
        )
        decision_log.add(
            agent="TuningAgent",
            action="传统算法粗搜与细搜",
            reason="根据当前评价目标对候选参数排序，并围绕 Top 结果继续细搜。",
            details={
                "objective": self.objective,
                "evaluated_count": len(raw_results),
                "valid_count": len(results),
                "best_algorithm": results[0].algorithm if results else None,
                "best_score": round(results[0].score, 4) if results else None,
            },
        )
        decision_log.add(
            agent="ScoringPolicyAgent",
            action="生成自适应综合评分权重",
            reason=scoring_policy.reason,
            details={
                "used_llm": scoring_policy.used_llm,
                "error": scoring_policy.error,
                "weights": {
                    name: round(value, 4)
                    for name, value in scoring_policy.weights.items()
                },
            },
        )
        if strategy_plan is not None:
            decision_log.add(
                agent="LLMStrategyAgent",
                action="生成下一轮调参建议",
                reason=strategy_plan.reason,
                details={
                    "used_llm": strategy_plan.used_llm,
                    "error": strategy_plan.error,
                    "suggestions": [
                        {
                            "algorithm": suggestion.algorithm,
                            "params": suggestion.params,
                            "reason": suggestion.reason,
                            "source": suggestion.source,
                            "base_score": suggestion.base_score,
                        }
                        for suggestion in strategy_plan.suggestions
                    ],
                },
            )
        deep_learning_used = False
        deep_learning_reason = "未启用深度学习后端。"

        deep_policy = DeepPolicyAgent(
            include_deep_learning=self.include_deep_learning,
            policy=self.deep_learning_policy,
            max_combinations=self.max_combinations,
        )
        deep_decision = deep_policy.decide(results)
        self._emit(
            agent="DeepPolicyAgent",
            action="判断是否启用 SuperPoint/SuperGlue",
            should_run=deep_decision.should_run,
            deep_budget=deep_decision.budget,
            reason=deep_decision.reason,
        )
        deep_learning_reason = deep_decision.reason
        decision_log.add(
            agent="DeepPolicyAgent",
            action="判断是否启用深度学习匹配",
            reason=deep_decision.reason,
            details={
                "policy": self.deep_learning_policy,
                "should_run": deep_decision.should_run,
                "deep_budget": deep_decision.budget,
            },
        )

        if deep_decision.should_run:
            deep_learning_used = True
            self._emit(
                agent="AlgorithmSelectorAgent",
                action="生成深度学习匹配候选",
                phase="deep",
            )
            deep_candidates = selector.deep_learning_candidates(image_profile)
            decision_log.add(
                agent="AlgorithmSelectorAgent",
                action="生成深度学习候选",
                reason="DeepPolicyAgent 判定传统算法质量不足或用户要求对比深度学习。",
                details={
                    "candidate_count": len(deep_candidates),
                    "budget": deep_decision.budget,
                    "algorithms": sorted({candidate.name for candidate in deep_candidates}),
                },
            )
            deep_raw_results = TuningAgent(
                max_combinations=deep_decision.budget,
                objective=self.objective,
                progress_callback=self.progress_callback,
            ).run(
                object_image=object_image_used,
                scene_image=scene_image,
                candidates=deep_candidates,
                object_mask=object_mask,
            )
            for result in deep_raw_results:
                result.search_phase = f"deep-{result.search_phase}"
                result.search_step += len(raw_results)
            raw_results.extend(deep_raw_results)
            results = EvaluatorAgent(
                objective=self.objective,
                custom_weights=scoring_policy.weights,
            ).score(raw_results)
            self._emit(
                agent="EvaluatorAgent",
                action="纳入深度学习结果后重新排序",
                phase="deep",
                evaluated_count=len(raw_results),
                best_algorithm=results[0].algorithm if results else None,
                best_score=round(results[0].score, 4) if results else None,
            )
            decision_log.add(
                agent="TuningAgent",
                action="运行深度学习候选",
                reason="将 SuperPoint/SuperGlue 结果纳入同一评价体系。",
                details={
                    "evaluated_count": len(deep_raw_results),
                    "valid_deep_count": len([result for result in deep_raw_results if result.error is None]),
                },
            )

        failed_results = [result for result in raw_results if result.error is not None]
        best_result = results[0] if results else None
        self._emit(
            agent="EvaluatorAgent",
            action="选择最终最优结果",
            evaluated_count=len(raw_results),
            best_algorithm=best_result.algorithm if best_result else None,
            best_score=round(best_result.score, 4) if best_result else None,
        )
        decision_log.add(
            agent="EvaluatorAgent",
            action="选择最终最优结果",
            reason="按当前评价目标对所有有效结果进行综合评分排序。",
            details={
                "objective": self.objective,
                "best_algorithm": best_result.algorithm if best_result else None,
                "best_score": round(best_result.score, 4) if best_result else None,
                "failed_count": len(failed_results),
            },
        )
        return PipelineOutput(
            image_profile=image_profile,
            results=results,
            failed_results=failed_results,
            best_result=best_result,
            objective=self.objective,
            deep_learning_used=deep_learning_used,
            deep_learning_reason=deep_learning_reason,
            strategy_plan=strategy_plan,
            scoring_policy=scoring_policy,
            decision_log=decision_log,
            object_image_used=object_image_used,
            object_crop_box=crop_box,
            object_mask_used=object_mask,
            object_mask_box=mask_box,
            object_mask_area_ratio=mask_area_ratio,
        )

    def _emit(self, **event: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

    def run_and_save(
        self,
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        object_path: Path,
        scene_path: Path,
        results_dir: Path,
    ) -> PipelineOutput:
        output = self.run(object_image, scene_image)
        reporter = ReporterAgent(results_dir)
        if output.best_result is not None:
            reporter.write_best_visualization(
                object_image=output.object_image_used,
                scene_image=scene_image,
                result=output.best_result,
                object_mask=output.object_mask_used,
            )
        reporter.write_csv(output.results)
        reporter.write_decision_log(output.decision_log)
        reporter.write_report(
            object_path=object_path,
            scene_path=scene_path,
            image_profile=output.image_profile,
            results=output.results,
            best_result=output.best_result,
        )

        return output

    def _auto_mask_object_image(
        self,
        image: cv2.Mat,
    ) -> tuple[cv2.Mat | None, tuple[int, int, int, int] | None]:
        height, width = image.shape[:2]
        if height < 40 or width < 40:
            return None, None

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        border_width = max(4, min(width, height) // 60)
        border = np.concatenate(
            [
                image_lab[:border_width, :, :].reshape(-1, 3),
                image_lab[-border_width:, :, :].reshape(-1, 3),
                image_lab[:, :border_width, :].reshape(-1, 3),
                image_lab[:, -border_width:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border.astype(np.float32), axis=0)
        border_distance = np.linalg.norm(border.astype(np.float32) - background, axis=1)
        color_distance = np.linalg.norm(image_lab.astype(np.float32) - background, axis=2)
        threshold = max(12.0, float(np.percentile(border_distance, 90) + 12.0))

        mask = (color_distance > threshold).astype(np.uint8) * 255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
        mask = cv2.bitwise_or(mask, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        box = self._best_component_box(mask)
        if box is None:
            return None, None
        cleaned = np.zeros(mask.shape, dtype=np.uint8)
        x, y, box_width, box_height = box
        component = mask[y : y + box_height, x : x + box_width]
        cleaned[y : y + box_height, x : x + box_width] = component
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        area_ratio = self._mask_area_ratio(cleaned)
        box_area_ratio = (box_width * box_height) / float(width * height)
        if area_ratio < 0.03 or area_ratio > 0.85 or box_area_ratio > 0.92:
            return None, None

        return cleaned, box

    def _mask_area_ratio(self, mask: cv2.Mat | None) -> float:
        if mask is None:
            return 0.0
        return float(np.count_nonzero(mask) / mask.size)

    def _auto_crop_object_image(
        self,
        image: cv2.Mat,
    ) -> tuple[cv2.Mat, tuple[int, int, int, int] | None]:
        height, width = image.shape[:2]
        if height < 40 or width < 40:
            return image, None

        color_crop = self._crop_by_background_color(image)
        if color_crop is not None:
            return color_crop

        edge_crop = self._crop_by_edges(image)
        if edge_crop is not None:
            return edge_crop

        return image, None

    def _crop_by_background_color(
        self,
        image: cv2.Mat,
    ) -> tuple[cv2.Mat, tuple[int, int, int, int]] | None:
        height, width = image.shape[:2]
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        border = np.concatenate(
            [
                image_lab[:6, :, :].reshape(-1, 3),
                image_lab[-6:, :, :].reshape(-1, 3),
                image_lab[:, :6, :].reshape(-1, 3),
                image_lab[:, -6:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border.astype(np.float32), axis=0)
        border_distance = np.linalg.norm(border.astype(np.float32) - background, axis=1)
        color_distance = np.linalg.norm(image_lab.astype(np.float32) - background, axis=2)
        threshold = max(18.0, float(np.percentile(border_distance, 92) + 16.0))

        mask = (color_distance > threshold).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        crop_box = self._best_component_box(mask)
        if crop_box is None:
            return None

        return self._crop_with_padding(image, crop_box, padding_ratio=0.012)

    def _crop_by_edges(
        self,
        image: cv2.Mat,
    ) -> tuple[cv2.Mat, tuple[int, int, int, int]] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 180)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        crop_box = self._best_component_box(mask)
        if crop_box is None:
            return None

        return self._crop_with_padding(image, crop_box, padding_ratio=0.01)

    def _best_component_box(self, mask: cv2.Mat) -> tuple[int, int, int, int] | None:
        height, width = mask.shape[:2]
        image_area = float(width * height)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        center = np.array([width / 2.0, height / 2.0])

        best_score = 0.0
        best_box = None
        for label in range(1, num_labels):
            x, y, component_width, component_height, area = stats[label]
            area_ratio = float(area) / image_area
            if area_ratio < 0.015 or area_ratio > 0.75:
                continue
            box_area_ratio = float(component_width * component_height) / image_area
            if box_area_ratio < 0.04 or box_area_ratio > 0.85:
                continue

            aspect = component_width / max(component_height, 1)
            if aspect < 0.08 or aspect > 12.0:
                continue

            component_center = np.array(centroids[label])
            center_distance = np.linalg.norm((component_center - center) / np.array([width, height]))
            center_score = 1.0 - min(center_distance * 2.0, 1.0)
            fill_score = min(float(area) / max(component_width * component_height, 1) * 2.0, 1.0)
            area_score = min(box_area_ratio / 0.28, 1.0)
            score = 0.45 * area_score + 0.35 * center_score + 0.20 * fill_score

            if score > best_score:
                best_score = score
                best_box = (int(x), int(y), int(component_width), int(component_height))

        return best_box

    def _crop_with_padding(
        self,
        image: cv2.Mat,
        crop_box: tuple[int, int, int, int],
        padding_ratio: float,
    ) -> tuple[cv2.Mat, tuple[int, int, int, int]] | None:
        height, width = image.shape[:2]
        x, y, crop_width, crop_height = crop_box
        padding = max(2, int(min(width, height) * padding_ratio))
        x0 = max(x - padding, 0)
        y0 = max(y - padding, 0)
        x1 = min(x + crop_width + padding, width)
        y1 = min(y + crop_height + padding, height)

        crop_area = float((x1 - x0) * (y1 - y0))
        image_area = float(width * height)
        if crop_area > image_area * 0.90 or crop_area < image_area * 0.04:
            return None

        return image[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)
