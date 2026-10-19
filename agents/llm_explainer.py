from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from agents.decision_log import DecisionLog
from agents.settings import get_deepseek_api_key
from agents.types import ImageProfile, MatchResult


class LLMExplanationAgent:
    """Generates a final experiment explanation with DeepSeek or a local fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: int = 60,
        max_tokens: int = 1800,
    ) -> None:
        self.api_key = api_key or get_deepseek_api_key()
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def generate(
        self,
        image_profile: ImageProfile,
        results: list[MatchResult],
        failed_results: list[MatchResult],
        best_result: MatchResult | None,
        deep_learning_reason: str,
        decision_log: DecisionLog | None = None,
    ) -> str:
        context = self._build_context(
            image_profile=image_profile,
            results=results,
            failed_results=failed_results,
            best_result=best_result,
            deep_learning_reason=deep_learning_reason,
            decision_log=decision_log,
        )

        if not self.api_key:
            return self._fallback_summary(context, reason="未配置 DEEPSEEK_API_KEY 或页面 API Key。")

        try:
            return self._call_deepseek(context)
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_summary(context, reason=f"DeepSeek 调用失败：{exc}")

    def _call_deepseek(self, context: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是一个计算机视觉课程大作业的实验报告解释 Agent。"
                        "请根据结构化指标生成中文总结，重点解释算法选择、调参过程、"
                        "匹配质量、深度学习策略和可改进方向。不要编造没有给出的数据。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请用 4 个小节生成最终实验总结：\n"
                        "1. Agent 决策过程\n"
                        "2. 最优算法与参数解释\n"
                        "3. 结果可靠性分析\n"
                        "4. 后续改进方向\n\n"
                        f"实验数据 JSON：\n{json.dumps(context, ensure_ascii=False)}"
                    ),
                },
            ],
            "temperature": 0.3,
            "max_tokens": self.max_tokens,
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
        choice = data["choices"][0]
        content = choice["message"]["content"].strip()
        if choice.get("finish_reason") == "length":
            content += (
                "\n\n> 注意：本次 LLM 总结达到输出长度上限，内容可能未完全结束。"
                "可以提高 `LLMExplanationAgent(max_tokens=...)` 后重新生成。"
            )
        return content

    def _build_context(
        self,
        image_profile: ImageProfile,
        results: list[MatchResult],
        failed_results: list[MatchResult],
        best_result: MatchResult | None,
        deep_learning_reason: str,
        decision_log: DecisionLog | None,
    ) -> dict[str, Any]:
        return {
            "image_profile": {
                "object_size": image_profile.object_size,
                "scene_size": image_profile.scene_size,
                "object_edge_density": round(image_profile.object_edge_density, 6),
                "scene_edge_density": round(image_profile.scene_edge_density, 6),
                "object_texture_score": round(image_profile.object_texture_score, 3),
                "scene_texture_score": round(image_profile.scene_texture_score, 3),
            },
            "deep_learning_policy": deep_learning_reason,
            "decision_log": decision_log.rows() if decision_log is not None else [],
            "best_result": self._result_brief(best_result) if best_result is not None else None,
            "top_results": [self._result_brief(result) for result in results[:5]],
            "failed_algorithms": [
                {
                    "algorithm": result.algorithm,
                    "phase": result.search_phase,
                    "error": result.error,
                }
                for result in failed_results[:8]
            ],
        }

    def _result_brief(self, result: MatchResult) -> dict[str, Any]:
        return {
            "algorithm": result.algorithm,
            "search_phase": result.search_phase,
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
            "confidence_reason": result.confidence_reason,
            "visualization_box_type": result.visualization_box_type,
            "visualization_box_reason": result.visualization_box_reason,
            "runtime_ms": round(result.runtime_ms, 3),
            "score_breakdown": {
                name: round(value, 6) for name, value in result.score_breakdown.items()
            },
        }

    def _fallback_summary(self, context: dict[str, Any], reason: str) -> str:
        best = context.get("best_result")
        if best is None:
            return f"### LLM 总结暂不可用\n\n{reason}\n\n本次实验没有产生有效匹配结果。"

        failed = context.get("failed_algorithms", [])
        failed_text = "；".join(
            f"{item['algorithm']}：{item['error']}" for item in failed[:3] if item.get("error")
        )
        if not failed_text:
            failed_text = "无主要失败候选。"

        return (
            "### Agent 决策过程\n\n"
            f"{reason} 当前采用本地模板总结。系统先分析图像边缘、纹理和尺寸，"
            "再生成传统算法候选并执行粗搜、细搜和早停策略。"
            f"{context['deep_learning_policy']}\n\n"
            "### 最优算法与参数解释\n\n"
            f"最终选择 `{best['algorithm']}`，综合得分为 `{best['score']}`，"
            f"RANSAC 内点数为 `{best['inliers']}`，内点率为 `{best['inlier_ratio']}`，"
            f"重投影误差为 `{best['reprojection_error']}`。参数为 `{best['params']}`。\n\n"
            "### 结果可靠性分析\n\n"
            f"覆盖度为 `{best['match_coverage']}`，框合理性为 `{best['homography_quality']}`。"
            "覆盖度越高，说明匹配点越分散，投影框越不容易由局部区域外推导致偏移。\n\n"
            "### 后续改进方向\n\n"
            f"可继续增加更多样例、优化目标图自动裁剪、调节深度学习按需启用阈值。"
            f"跳过或失败候选：{failed_text}"
        )
