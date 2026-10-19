from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2


@dataclass(frozen=True)
class ImageProfile:
    object_size: tuple[int, int]
    scene_size: tuple[int, int]
    object_edge_density: float
    scene_edge_density: float
    object_texture_score: float
    scene_texture_score: float


@dataclass(frozen=True)
class AlgorithmCandidate:
    name: str
    params: dict[str, Any]


@dataclass
class MatchResult:
    algorithm: str
    params: dict[str, Any]
    search_phase: str = "initial"
    search_step: int = 0
    keypoints_object: list[cv2.KeyPoint] = field(default_factory=list)
    keypoints_scene: list[cv2.KeyPoint] = field(default_factory=list)
    raw_matches: list[cv2.DMatch] = field(default_factory=list)
    good_matches: list[cv2.DMatch] = field(default_factory=list)
    inlier_matches: list[cv2.DMatch] = field(default_factory=list)
    homography: Any = None
    runtime_ms: float = 0.0
    avg_distance: float = 0.0
    reprojection_error: float | None = None
    match_coverage: float = 0.0
    projected_area_ratio: float | None = None
    projected_inside_ratio: float = 0.0
    homography_quality: float = 0.0
    evidence_score: float = 0.0
    spatial_distribution_score: float = 0.0
    confidence_level: str = "unknown"
    confidence_reason: str = ""
    visualization_box_type: str = "not_drawn"
    visualization_box_reason: str = ""
    scene_inlier_bbox: tuple[int, int, int, int] | None = None
    localization_status: str = "unknown"
    error: str | None = None
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def num_keypoints_object(self) -> int:
        return len(self.keypoints_object)

    @property
    def num_keypoints_scene(self) -> int:
        return len(self.keypoints_scene)

    @property
    def num_raw_matches(self) -> int:
        return len(self.raw_matches)

    @property
    def num_good_matches(self) -> int:
        return len(self.good_matches)

    @property
    def num_inliers(self) -> int:
        return len(self.inlier_matches)

    @property
    def inlier_ratio(self) -> float:
        if not self.good_matches:
            return 0.0
        return self.num_inliers / len(self.good_matches)
