from __future__ import annotations

import os
import sys
import time
from typing import Any

import cv2
import numpy as np

from agents.types import MatchResult


class FastBriskFeature:
    """FAST detects corners quickly, while BRISK computes binary descriptors."""

    def __init__(
        self,
        threshold: int,
        nonmax_suppression: bool,
        nfeatures: int,
    ) -> None:
        self.detector = cv2.FastFeatureDetector_create(
            threshold=threshold,
            nonmaxSuppression=nonmax_suppression,
        )
        self.descriptor = cv2.BRISK_create()
        self.nfeatures = nfeatures

    def detectAndCompute(self, image: cv2.Mat, mask: cv2.Mat | None):
        keypoints = self.detector.detect(image, mask)
        keypoints = sorted(keypoints, key=lambda item: item.response, reverse=True)
        if self.nfeatures > 0:
            keypoints = keypoints[: self.nfeatures]
        return self.descriptor.compute(image, keypoints)


class FeatureMatcher:
    def __init__(self) -> None:
        self._superglue_models: dict[tuple[str, str], Any] = {}

    def run(
        self,
        algorithm: str,
        params: dict[str, Any],
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
        object_mask: cv2.Mat | None = None,
    ) -> MatchResult:
        started = time.perf_counter()
        result = MatchResult(algorithm=algorithm, params=params)

        try:
            if algorithm == "SUPERPOINT_SUPERGLUE":
                self._run_superpoint_superglue(
                    result=result,
                    params=params,
                    object_image=object_image,
                    scene_image=scene_image,
                )
                return result

            gray_object = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
            gray_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
            detector, norm_type = self._create_detector(algorithm, params)

            kp_object, desc_object = detector.detectAndCompute(gray_object, object_mask)
            kp_scene, desc_scene = detector.detectAndCompute(gray_scene, None)
            result.keypoints_object = kp_object or []
            result.keypoints_scene = kp_scene or []

            if desc_object is None or desc_scene is None:
                result.error = "No descriptors found."
                return result

            knn_matches = self._match_descriptors(
                desc_object=desc_object,
                desc_scene=desc_scene,
                norm_type=norm_type,
                matcher=params.get("matcher", "BF"),
            )

            ratio = float(params.get("ratio", 0.75))
            result.raw_matches = [m for pair in knn_matches for m in pair[:1]]
            result.good_matches = self._ratio_filter(knn_matches, ratio)
            result.avg_distance = self._avg_distance(result.good_matches)
            self._apply_ransac(result, params, object_image.shape, scene_image.shape)
            return result
        except Exception as exc:  # Keep one bad parameter set from stopping the search.
            result.error = str(exc)
            return result
        finally:
            result.runtime_ms = (time.perf_counter() - started) * 1000

    def _create_detector(
        self, algorithm: str, params: dict[str, Any]
    ) -> tuple[cv2.Feature2D, int]:
        if algorithm == "SIFT":
            return (
                cv2.SIFT_create(
                    nfeatures=int(params.get("nfeatures", 0)),
                    contrastThreshold=float(params.get("contrastThreshold", 0.04)),
                    edgeThreshold=float(params.get("edgeThreshold", 10)),
                ),
                cv2.NORM_L2,
            )

        if algorithm == "SURF":
            xfeatures2d = getattr(cv2, "xfeatures2d", None)
            if xfeatures2d is None or not hasattr(xfeatures2d, "SURF_create"):
                raise RuntimeError("SURF requires opencv-contrib-python with nonfree modules.")
            return (
                xfeatures2d.SURF_create(
                    hessianThreshold=float(params.get("hessianThreshold", 400)),
                    nOctaves=int(params.get("nOctaves", 4)),
                    nOctaveLayers=int(params.get("nOctaveLayers", 3)),
                    extended=bool(params.get("extended", False)),
                    upright=bool(params.get("upright", False)),
                ),
                cv2.NORM_L2,
            )

        if algorithm == "ORB":
            return (
                cv2.ORB_create(
                    nfeatures=int(params.get("nfeatures", 500)),
                    scaleFactor=float(params.get("scaleFactor", 1.2)),
                    nlevels=int(params.get("nlevels", 8)),
                ),
                cv2.NORM_HAMMING,
            )

        if algorithm == "FAST_BRISK":
            return (
                FastBriskFeature(
                    threshold=int(params.get("fastThreshold", 20)),
                    nonmax_suppression=bool(params.get("nonmaxSuppression", True)),
                    nfeatures=int(params.get("nfeatures", 1000)),
                ),
                cv2.NORM_HAMMING,
            )

        if algorithm == "AKAZE":
            return (
                cv2.AKAZE_create(
                    threshold=float(params.get("threshold", 0.001)),
                    nOctaves=int(params.get("nOctaves", 4)),
                ),
                cv2.NORM_HAMMING,
            )

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _run_superpoint_superglue(
        self,
        result: MatchResult,
        params: dict[str, Any],
        object_image: cv2.Mat,
        scene_image: cv2.Mat,
    ) -> None:
        torch = self._import_torch()
        matching = self._load_superglue_model(params, torch)
        device = next(matching.parameters()).device

        gray_object = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
        gray_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
        tensor_object = self._frame_to_tensor(gray_object, torch, device)
        tensor_scene = self._frame_to_tensor(gray_scene, torch, device)

        with torch.no_grad():
            prediction = matching({"image0": tensor_object, "image1": tensor_scene})

        keypoints_object = prediction["keypoints0"][0].detach().cpu().numpy()
        keypoints_scene = prediction["keypoints1"][0].detach().cpu().numpy()
        matches = prediction["matches0"][0].detach().cpu().numpy()
        scores = prediction["matching_scores0"][0].detach().cpu().numpy()
        min_confidence = float(params.get("matchThreshold", 0.2))

        result.keypoints_object = [
            cv2.KeyPoint(float(point[0]), float(point[1]), 3) for point in keypoints_object
        ]
        result.keypoints_scene = [
            cv2.KeyPoint(float(point[0]), float(point[1]), 3) for point in keypoints_scene
        ]

        good_matches = []
        for query_index, train_index in enumerate(matches):
            if train_index < 0 or scores[query_index] < min_confidence:
                continue
            good_matches.append(
                cv2.DMatch(
                    _queryIdx=int(query_index),
                    _trainIdx=int(train_index),
                    _distance=float(1.0 - scores[query_index]),
                )
            )

        result.raw_matches = good_matches
        result.good_matches = good_matches
        result.avg_distance = self._avg_distance(good_matches)
        self._apply_ransac(result, params, object_image.shape, scene_image.shape)

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("SuperPoint/SuperGlue requires PyTorch.") from exc
        return torch

    def _load_superglue_model(self, params: dict[str, Any], torch):
        repo_path = str(
            params.get("repoPath")
            or os.environ.get("SUPERGLUE_REPO")
            or "third_party/SuperGluePretrainedNetwork"
        )
        repo_path = os.path.abspath(repo_path)
        if not os.path.isdir(repo_path):
            raise RuntimeError(
                "SuperPoint/SuperGlue repo not found. Set SUPERGLUE_REPO or place it at "
                "third_party/SuperGluePretrainedNetwork."
            )

        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from models.matching import Matching
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import models.matching from the SuperGlue repository."
            ) from exc

        weights = str(params.get("weights", "outdoor"))
        device_name = str(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        cache_key = (
            repo_path,
            weights,
            str(params.get("nmsRadius", 4)),
            str(params.get("keypointThreshold", 0.005)),
            str(params.get("maxKeypoints", 1024)),
            str(params.get("sinkhornIterations", 20)),
            str(params.get("matchThreshold", 0.2)),
            device_name,
        )
        if cache_key not in self._superglue_models:
            config = {
                "superpoint": {
                    "nms_radius": int(params.get("nmsRadius", 4)),
                    "keypoint_threshold": float(params.get("keypointThreshold", 0.005)),
                    "max_keypoints": int(params.get("maxKeypoints", 1024)),
                },
                "superglue": {
                    "weights": weights,
                    "sinkhorn_iterations": int(params.get("sinkhornIterations", 20)),
                    "match_threshold": float(params.get("matchThreshold", 0.2)),
                },
            }
            self._superglue_models[cache_key] = Matching(config).eval().to(device_name)

        return self._superglue_models[cache_key]

    def _frame_to_tensor(self, image: cv2.Mat, torch, device):
        normalized = image.astype(np.float32) / 255.0
        return torch.from_numpy(normalized)[None, None].to(device)

    def _match_descriptors(
        self,
        desc_object: np.ndarray,
        desc_scene: np.ndarray,
        norm_type: int,
        matcher: str,
    ) -> list[list[cv2.DMatch]]:
        if matcher == "FLANN" and norm_type == cv2.NORM_L2:
            index_params = {"algorithm": 1, "trees": 5}
            search_params = {"checks": 50}
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            return flann.knnMatch(desc_object, desc_scene, k=2)

        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        return bf.knnMatch(desc_object, desc_scene, k=2)

    def _ratio_filter(
        self, knn_matches: list[list[cv2.DMatch]], ratio: float
    ) -> list[cv2.DMatch]:
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < ratio * second.distance:
                good_matches.append(first)
        return good_matches

    def _avg_distance(self, matches: list[cv2.DMatch]) -> float:
        if not matches:
            return 0.0
        return float(sum(match.distance for match in matches) / len(matches))

    def _apply_ransac(
        self,
        result: MatchResult,
        params: dict[str, Any],
        object_shape: tuple[int, ...],
        scene_shape: tuple[int, ...],
    ) -> None:
        if len(result.good_matches) < 4:
            return

        src_points = np.float32(
            [result.keypoints_object[m.queryIdx].pt for m in result.good_matches]
        ).reshape(-1, 1, 2)
        dst_points = np.float32(
            [result.keypoints_scene[m.trainIdx].pt for m in result.good_matches]
        ).reshape(-1, 1, 2)

        threshold = float(params.get("ransacReprojThreshold", 5.0))
        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
        result.homography = homography

        if mask is None:
            return

        inlier_mask = mask.ravel().astype(bool)
        result.inlier_matches = [
            match for match, is_inlier in zip(result.good_matches, inlier_mask) if is_inlier
        ]
        result.reprojection_error = self._reprojection_error(
            src_points=src_points[inlier_mask],
            dst_points=dst_points[inlier_mask],
            homography=homography,
        )
        self._measure_spatial_quality(
            result=result,
            object_shape=object_shape,
            scene_shape=scene_shape,
        )

    def _reprojection_error(
        self, src_points: np.ndarray, dst_points: np.ndarray, homography: np.ndarray | None
    ) -> float | None:
        if homography is None or len(src_points) == 0:
            return None
        projected = cv2.perspectiveTransform(src_points, homography)
        errors = np.linalg.norm(projected.reshape(-1, 2) - dst_points.reshape(-1, 2), axis=1)
        return float(errors.mean())

    def _measure_spatial_quality(
        self,
        result: MatchResult,
        object_shape: tuple[int, ...],
        scene_shape: tuple[int, ...],
    ) -> None:
        object_height, object_width = object_shape[:2]
        scene_height, scene_width = scene_shape[:2]
        object_area = float(max(object_width * object_height, 1))
        scene_area = float(max(scene_width * scene_height, 1))

        if result.inlier_matches:
            object_points = np.float32(
                [result.keypoints_object[m.queryIdx].pt for m in result.inlier_matches]
            )
            scene_points = np.float32(
                [result.keypoints_scene[m.trainIdx].pt for m in result.inlier_matches]
            )
            x_min, y_min = object_points.min(axis=0)
            x_max, y_max = object_points.max(axis=0)
            bbox_area = max(float((x_max - x_min) * (y_max - y_min)), 0.0)
            result.match_coverage = float(np.clip(bbox_area / object_area, 0.0, 1.0))
            sx_min, sy_min = scene_points.min(axis=0)
            sx_max, sy_max = scene_points.max(axis=0)
            result.scene_inlier_bbox = (
                int(max(sx_min, 0)),
                int(max(sy_min, 0)),
                int(max(sx_max - sx_min, 1)),
                int(max(sy_max - sy_min, 1)),
            )
            result.spatial_distribution_score = self._spatial_distribution_score(
                points=object_points,
                width=object_width,
                height=object_height,
            )

        if result.homography is None:
            result.localization_status = "no_homography"
            self._measure_evidence_score(result)
            return

        corners = np.float32(
            [[0, 0], [object_width, 0], [object_width, object_height], [0, object_height]]
        ).reshape(-1, 1, 2)
        try:
            projected = cv2.perspectiveTransform(corners, result.homography).reshape(-1, 2)
        except cv2.error:
            return

        if not np.isfinite(projected).all():
            result.localization_status = "invalid_projection"
            return

        projected_contour = projected.astype(np.float32).reshape(-1, 1, 2)
        projected_area = abs(float(cv2.contourArea(projected_contour)))
        if projected_area <= 1.0:
            result.localization_status = "invalid_area"
            return

        result.projected_area_ratio = projected_area / object_area
        inside_x = (projected[:, 0] >= 0) & (projected[:, 0] <= scene_width)
        inside_y = (projected[:, 1] >= 0) & (projected[:, 1] <= scene_height)
        result.projected_inside_ratio = float(np.count_nonzero(inside_x & inside_y) / 4.0)

        convex = cv2.isContourConvex(projected_contour)
        scene_fill = projected_area / scene_area
        area_ratio = max(result.projected_area_ratio, 1e-6)
        area_score = 1.0 - min(abs(np.log(area_ratio)) / np.log(20.0), 1.0)
        scene_score = 1.0 if 0.002 <= scene_fill <= 0.95 else 0.25
        convex_score = 1.0 if convex else 0.0
        result.homography_quality = float(
            np.clip(
                0.45 * area_score
                + 0.30 * result.projected_inside_ratio
                + 0.15 * convex_score
                + 0.10 * scene_score,
                0.0,
                1.0,
            )
        )
        if result.homography_quality < 0.15:
            result.localization_status = "unreliable_homography"
        elif result.match_coverage < 0.25:
            result.localization_status = "low_coverage"
        else:
            result.localization_status = "reliable"
        self._measure_evidence_score(result)

    def _spatial_distribution_score(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ) -> float:
        if len(points) == 0:
            return 0.0

        grid_cols = 3
        grid_rows = 3
        xs = np.clip((points[:, 0] / max(width, 1) * grid_cols).astype(int), 0, grid_cols - 1)
        ys = np.clip((points[:, 1] / max(height, 1) * grid_rows).astype(int), 0, grid_rows - 1)
        occupied = len(set(zip(xs.tolist(), ys.tolist())))
        grid_score = occupied / float(grid_cols * grid_rows)

        if len(points) < 3:
            shape_score = 0.0
        else:
            centered = points - points.mean(axis=0)
            _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
            if len(singular_values) < 2 or singular_values[0] <= 1e-6:
                shape_score = 0.0
            else:
                shape_score = float(np.clip(singular_values[1] / singular_values[0], 0.0, 1.0))

        count_score = min(len(points) / 24.0, 1.0)
        return float(np.clip(0.45 * grid_score + 0.35 * shape_score + 0.20 * count_score, 0.0, 1.0))

    def _measure_evidence_score(self, result: MatchResult) -> None:
        inlier_count_score = min(result.num_inliers / 24.0, 1.0)
        match_count_score = min(result.num_good_matches / 40.0, 1.0)
        result.evidence_score = float(
            np.clip(
                0.30 * inlier_count_score
                + 0.20 * match_count_score
                + 0.20 * result.inlier_ratio
                + 0.15 * result.match_coverage
                + 0.15 * result.spatial_distribution_score,
                0.0,
                1.0,
            )
        )
