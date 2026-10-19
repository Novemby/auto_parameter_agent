from __future__ import annotations

from dataclasses import dataclass

from agents.types import MatchResult


@dataclass(frozen=True)
class DeepPolicyDecision:
    should_run: bool
    reason: str
    budget: int


class DeepPolicyAgent:
    """Decides whether the deep-learning matcher is worth running."""

    def __init__(
        self,
        include_deep_learning: bool,
        policy: str,
        max_combinations: int | None,
    ) -> None:
        self.include_deep_learning = include_deep_learning
        self.policy = policy
        self.max_combinations = max_combinations

    def decide(self, traditional_results: list[MatchResult]) -> DeepPolicyDecision:
        if not self.include_deep_learning:
            return DeepPolicyDecision(
                should_run=False,
                reason="侧边栏未勾选 SuperPoint/SuperGlue。",
                budget=0,
            )

        if self.policy == "从不启用":
            return DeepPolicyDecision(
                should_run=False,
                reason="用户选择只运行传统算法。",
                budget=0,
            )

        if self.policy == "总是启用":
            return DeepPolicyDecision(
                should_run=True,
                reason="用户选择总是启用深度学习后端。",
                budget=self._budget(),
            )

        should_run, reason = self._quality_gate(traditional_results)
        return DeepPolicyDecision(
            should_run=should_run,
            reason=reason,
            budget=self._budget() if should_run else 0,
        )

    def _quality_gate(self, results: list[MatchResult]) -> tuple[bool, str]:
        if not results:
            return True, "传统算法没有得到有效结果，启用 SuperPoint/SuperGlue 补充匹配。"

        best = results[0]
        if best.score < 0.55:
            return True, f"传统最优综合得分较低（{best.score:.4f}），启用深度学习补充。"
        if best.inlier_ratio < 0.35:
            return True, f"传统最优内点率较低（{best.inlier_ratio:.4f}），启用深度学习补充。"
        if best.match_coverage < 0.30:
            return True, f"传统最优覆盖度较低（{best.match_coverage:.4f}），启用深度学习补充。"
        if best.homography_quality < 0.70:
            return True, f"传统最优框合理性较低（{best.homography_quality:.4f}），启用深度学习补充。"
        if best.reprojection_error is not None and best.reprojection_error > 4.0:
            return True, f"传统最优重投影误差偏高（{best.reprojection_error:.4f}），启用深度学习补充。"

        return False, "传统算法结果已满足质量阈值，跳过 SuperPoint/SuperGlue 以节省时间。"

    def _budget(self) -> int:
        return 2 if self.max_combinations is not None else 3
