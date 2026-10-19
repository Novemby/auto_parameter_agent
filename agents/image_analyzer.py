from __future__ import annotations

import cv2
import numpy as np

from agents.types import ImageProfile


class ImageAnalyzerAgent:
    """Computes simple image statistics used by the selector agent."""

    def analyze(self, object_image: cv2.Mat, scene_image: cv2.Mat) -> ImageProfile:
        return ImageProfile(
            object_size=self._size(object_image),
            scene_size=self._size(scene_image),
            object_edge_density=self._edge_density(object_image),
            scene_edge_density=self._edge_density(scene_image),
            object_texture_score=self._texture_score(object_image),
            scene_texture_score=self._texture_score(scene_image),
        )

    def _size(self, image: cv2.Mat) -> tuple[int, int]:
        height, width = image.shape[:2]
        return width, height

    def _edge_density(self, image: cv2.Mat) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return float(np.count_nonzero(edges) / edges.size)

    def _texture_score(self, image: cv2.Mat) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
