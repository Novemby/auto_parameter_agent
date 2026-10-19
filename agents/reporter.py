from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from agents.decision_log import DecisionLog
from agents.types import ImageProfile, MatchResult


class ReporterAgent:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, results: list[MatchResult]) -> Path:
        path = self.results_dir / "all_results.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rank",
                    "search_phase",
                    "search_step",
                    "algorithm",
                    "score",
                    "num_keypoints_object",
                    "num_keypoints_scene",
                    "num_raw_matches",
                    "num_good_matches",
                    "num_inliers",
                    "inlier_ratio",
                    "avg_distance",
                    "reprojection_error",
                    "match_coverage",
                    "evidence_score",
                    "spatial_distribution_score",
                    "confidence_level",
                    "confidence_reason",
                    "visualization_box_type",
                    "visualization_box_reason",
                    "projected_area_ratio",
                    "projected_inside_ratio",
                    "homography_quality",
                    "runtime_ms",
                    "score_breakdown",
                    "params",
                ],
            )
            writer.writeheader()
            for index, result in enumerate(results, start=1):
                writer.writerow(
                    {
                        "rank": index,
                        "search_phase": result.search_phase,
                        "search_step": result.search_step,
                        "algorithm": result.algorithm,
                        "score": f"{result.score:.6f}",
                        "num_keypoints_object": result.num_keypoints_object,
                        "num_keypoints_scene": result.num_keypoints_scene,
                        "num_raw_matches": result.num_raw_matches,
                        "num_good_matches": result.num_good_matches,
                        "num_inliers": result.num_inliers,
                        "inlier_ratio": f"{result.inlier_ratio:.6f}",
                        "avg_distance": f"{result.avg_distance:.6f}",
                        "reprojection_error": self._format_optional(result.reprojection_error),
                        "match_coverage": f"{result.match_coverage:.6f}",
                        "evidence_score": f"{result.evidence_score:.6f}",
                        "spatial_distribution_score": f"{result.spatial_distribution_score:.6f}",
                        "confidence_level": result.confidence_level,
                        "confidence_reason": result.confidence_reason,
                        "visualization_box_type": result.visualization_box_type,
                        "visualization_box_reason": result.visualization_box_reason,
                        "projected_area_ratio": self._format_optional(result.projected_area_ratio),
                        "projected_inside_ratio": f"{result.projected_inside_ratio:.6f}",
                        "homography_quality": f"{result.homography_quality:.6f}",
                        "runtime_ms": f"{result.runtime_ms:.3f}",
                        "score_breakdown": json.dumps(result.score_breakdown, ensure_ascii=False),
                        "params": json.dumps(result.params, ensure_ascii=False),
                    }
                )
        return path

    def write_report(
        self,
        object_path: Path,
        scene_path: Path,
        image_profile: ImageProfile,
        results: list[MatchResult],
        best_result: MatchResult | None,
    ) -> Path:
        path = self.results_dir / "report.md"
        lines = [
            "# Feature Matching Auto Parameter Report",
            "",
            "## Input",
            "",
            f"- Object image: `{object_path}`",
            f"- Scene image: `{scene_path}`",
            "",
            "## Image Profile",
            "",
            f"- Object size: {image_profile.object_size}",
            f"- Scene size: {image_profile.scene_size}",
            f"- Object edge density: {image_profile.object_edge_density:.4f}",
            f"- Scene edge density: {image_profile.scene_edge_density:.4f}",
            f"- Object texture score: {image_profile.object_texture_score:.2f}",
            f"- Scene texture score: {image_profile.scene_texture_score:.2f}",
            "",
            "## Best Result",
            "",
        ]

        if best_result is None:
            lines.append("No valid result was produced.")
        else:
            lines.extend(
                [
                    f"- Algorithm: **{best_result.algorithm}**",
                    f"- Score: **{best_result.score:.4f}**",
                    f"- Params: `{json.dumps(best_result.params, ensure_ascii=False)}`",
                    f"- Object keypoints: {best_result.num_keypoints_object}",
                    f"- Scene keypoints: {best_result.num_keypoints_scene}",
                    f"- Good matches: {best_result.num_good_matches}",
                    f"- RANSAC inliers: {best_result.num_inliers}",
                    f"- Inlier ratio: {best_result.inlier_ratio:.4f}",
                    f"- Runtime: {best_result.runtime_ms:.2f} ms",
                    f"- Reprojection error: {self._format_optional(best_result.reprojection_error)}",
                    f"- Match coverage: {best_result.match_coverage:.4f}",
                    f"- Homography quality: {best_result.homography_quality:.4f}",
                    f"- Evidence score: {best_result.evidence_score:.4f}",
                    f"- Spatial distribution score: {best_result.spatial_distribution_score:.4f}",
                    f"- Confidence level: {best_result.confidence_level}",
                    f"- Confidence reason: {best_result.confidence_reason}",
                    f"- Visualization box type: {best_result.visualization_box_type}",
                    f"- Visualization box reason: {best_result.visualization_box_reason}",
                ]
            )

        lines.extend(["", "## Top 10 Results", ""])
        lines.append(
            "| Rank | Algorithm | Score | Good matches | Inliers | Inlier ratio | Runtime ms |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|")

        for index, result in enumerate(results[:10], start=1):
            lines.append(
                f"| {index} | {result.algorithm} | {result.score:.4f} | "
                f"{result.num_good_matches} | {result.num_inliers} | "
                f"{result.inlier_ratio:.4f} | {result.runtime_ms:.2f} |"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def write_llm_summary(self, summary: str) -> Path:
        path = self.results_dir / "llm_summary.md"
        path.write_text(summary.strip() + "\n", encoding="utf-8")
        return path

    def write_decision_log(self, decision_log: DecisionLog) -> Path:
        path = self.results_dir / "decision_log.json"
        payload = decision_log.rows()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_best_visualization(
        self,
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        result: MatchResult,
        object_mask: cv2.Mat | None = None,
    ) -> Path:
        path = self.results_dir / "best_match.jpg"
        matches = result.inlier_matches if result.inlier_matches else result.good_matches[:50]
        visualization = cv2.drawMatches(
            object_image,
            result.keypoints_object,
            scene_image,
            result.keypoints_scene,
            matches[:80],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        if (
            result.homography is not None
            and self._is_projected_box_reliable(result)
        ):
            result.visualization_box_type = (
                "homography_mask" if object_mask is not None else "homography"
            )
            result.visualization_box_reason = (
                "Homography 可靠，使用目标 mask 外接区域投影。"
                if object_mask is not None
                else "Homography 可靠，使用目标图矩形投影。"
            )
            visualization = self._draw_projected_object(
                visualization=visualization,
                object_image=object_image,
                scene_image=scene_image,
                result=result,
                object_mask=object_mask,
            )
        elif result.scene_inlier_bbox is not None:
            visualization = self._draw_low_confidence_bbox(
                visualization=visualization,
                object_image=object_image,
                result=result,
            )
        else:
            result.visualization_box_type = "not_drawn"
            result.visualization_box_reason = "没有足够可靠的匹配证据绘制定位框。"

        cv2.imwrite(str(path), visualization)
        return path

    def _draw_low_confidence_bbox(
        self,
        visualization: cv2.Mat,
        object_image: cv2.Mat,
        result: MatchResult,
    ) -> cv2.Mat:
        object_width = object_image.shape[1]
        result.visualization_box_type = "low_confidence"
        result.visualization_box_reason = (
            "Homography 可靠性不足，降级为场景内点包围框。"
        )
        x, y, width, height = result.scene_inlier_bbox
        top_left = (object_width + x, y)
        bottom_right = (object_width + x + width, y + height)
        cv2.rectangle(
            visualization,
            top_left,
            bottom_right,
            color=(0, 165, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            "low-confidence local match",
            (top_left[0], max(top_left[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return visualization

    def _is_projected_box_reliable(self, result: MatchResult) -> bool:
        if result.homography_quality < 0.30:
            return False
        if result.match_coverage < 0.25:
            return False
        if result.projected_inside_ratio < 0.50:
            return False

        min_inliers = self._dynamic_min_inliers(result)
        if result.num_inliers < min_inliers:
            return False

        if result.reprojection_error is not None and result.reprojection_error > 5.0:
            return False
        return True

    def _dynamic_min_inliers(self, result: MatchResult) -> int:
        min_inliers = 8

        strong_geometry = (
            result.homography_quality >= 0.78
            and result.match_coverage >= 0.50
            and result.projected_inside_ratio >= 0.75
            and result.inlier_ratio >= 0.60
            and result.reprojection_error is not None
            and result.reprojection_error <= 2.0
        )
        if strong_geometry:
            min_inliers = 6

        weak_spatial_evidence = (
            result.match_coverage < 0.35
            or result.homography_quality < 0.55
            or result.projected_inside_ratio < 0.75
        )
        if weak_spatial_evidence:
            min_inliers = max(min_inliers, 10)

        very_weak_evidence = (
            result.match_coverage < 0.25
            or result.homography_quality < 0.40
            or result.localization_status
            in {"invalid_projection", "invalid_area", "unreliable_homography"}
        )
        if very_weak_evidence:
            min_inliers = max(min_inliers, 12)

        return min_inliers

    def _draw_projected_object(
        self,
        visualization: cv2.Mat,
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        result: MatchResult,
        object_mask: cv2.Mat | None,
    ) -> cv2.Mat:
        height, width = object_image.shape[:2]
        source_box = self._mask_bbox(object_mask) if object_mask is not None else None
        projected_scene = self._project_object_corners(
            width=width,
            height=height,
            homography=result.homography,
            source_box=source_box,
        )
        if projected_scene is None:
            return visualization

        if self._should_use_stabilized_bbox(result, projected_scene):
            result.visualization_box_type = "stabilized"
            result.visualization_box_reason = (
                "Homography 可定位目标，但投影框相对内点分布存在外推风险，"
                "改用稳定框。"
            )
            return self._draw_stabilized_bbox(
                visualization=visualization,
                object_image=object_image,
                scene_image=scene_image,
                result=result,
            )

        projected = projected_scene.copy()
        projected[:, :, 0] += width
        points = np.int32(projected)
        cv2.polylines(
            visualization,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        return visualization

    def _project_object_corners(
        self,
        width: int,
        height: int,
        homography,
        source_box: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray | None:
        if source_box is None:
            corners = np.float32(
                [[0, 0], [width, 0], [width, height], [0, height]]
            ).reshape(-1, 1, 2)
        else:
            x, y, box_width, box_height = source_box
            corners = np.float32(
                [
                    [x, y],
                    [x + box_width, y],
                    [x + box_width, y + box_height],
                    [x, y + box_height],
                ]
            ).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(corners, homography)
        except cv2.error:
            return None
        if not np.isfinite(projected).all():
            return None
        return projected

    def _mask_bbox(self, mask: cv2.Mat) -> tuple[int, int, int, int] | None:
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, width, height = cv2.boundingRect(points)
        if width < 3 or height < 3:
            return None
        return int(x), int(y), int(width), int(height)

    def _should_use_stabilized_bbox(
        self,
        result: MatchResult,
        projected_scene: np.ndarray,
    ) -> bool:
        if result.scene_inlier_bbox is None:
            return False
        if result.homography_quality >= 0.82 and result.match_coverage >= 0.45:
            return False

        projected_box = self._bbox_from_points(projected_scene.reshape(-1, 2))
        inlier_box = result.scene_inlier_bbox
        if projected_box is None:
            return True

        projected_area = max(projected_box[2] * projected_box[3], 1)
        padded_inlier = self._padded_box(inlier_box, padding_ratio=0.18)
        inlier_area = max(padded_inlier[2] * padded_inlier[3], 1)
        area_expansion = projected_area / inlier_area
        center_distance = self._box_center_distance(projected_box, padded_inlier)

        return (
            area_expansion > 1.75
            or center_distance > 0.35
            or (result.match_coverage < 0.35 and result.homography_quality < 0.75)
        )

    def _draw_stabilized_bbox(
        self,
        visualization: cv2.Mat,
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        result: MatchResult,
    ) -> cv2.Mat:
        object_width = object_image.shape[1]
        scene_height, scene_width = scene_image.shape[:2]
        box = self._padded_box(result.scene_inlier_bbox, padding_ratio=0.22)
        x, y, width, height = self._clamp_box(box, scene_width, scene_height)
        top_left = (object_width + x, y)
        bottom_right = (object_width + x + width, y + height)
        cv2.rectangle(
            visualization,
            top_left,
            bottom_right,
            color=(0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            "stabilized match",
            (top_left[0], max(top_left[1] - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return visualization

    def _bbox_from_points(self, points: np.ndarray) -> tuple[int, int, int, int] | None:
        if len(points) == 0 or not np.isfinite(points).all():
            return None
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        return (
            int(round(x_min)),
            int(round(y_min)),
            int(round(max(x_max - x_min, 1))),
            int(round(max(y_max - y_min, 1))),
        )

    def _padded_box(
        self,
        box: tuple[int, int, int, int],
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        padding = int(round(max(width, height) * padding_ratio))
        return (
            x - padding,
            y - padding,
            width + padding * 2,
            height + padding * 2,
        )

    def _clamp_box(
        self,
        box: tuple[int, int, int, int],
        max_width: int,
        max_height: int,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = box
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + width, max_width)
        y1 = min(y + height, max_height)
        return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)

    def _box_center_distance(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        first_center = np.array([first[0] + first[2] / 2.0, first[1] + first[3] / 2.0])
        second_center = np.array([second[0] + second[2] / 2.0, second[1] + second[3] / 2.0])
        scale = max(first[2], first[3], second[2], second[3], 1)
        return float(np.linalg.norm(first_center - second_center) / scale)

    def _format_optional(self, value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"
