from __future__ import annotations

from agents.types import AlgorithmCandidate, ImageProfile


class AlgorithmSelectorAgent:
    """Selects candidate algorithms based on simple image statistics."""

    def select(
        self,
        profile: ImageProfile,
        include_deep_learning: bool = True,
    ) -> list[AlgorithmCandidate]:
        candidates: list[AlgorithmCandidate] = []
        candidates.extend(self._sift_candidates(profile))
        candidates.extend(self._surf_candidates(profile))
        candidates.extend(self._orb_candidates(profile))
        candidates.extend(self._fast_brisk_candidates(profile))
        candidates.extend(self._akaze_candidates(profile))
        if include_deep_learning:
            candidates.extend(self.deep_learning_candidates(profile))
        return candidates

    def deep_learning_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        return self._superpoint_superglue_candidates(profile)

    def _sift_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        texture = min(profile.object_texture_score, profile.scene_texture_score)
        if texture > 200:
            contrast_values = [0.02, 0.04, 0.08]
        else:
            contrast_values = [0.01, 0.02, 0.04]

        candidates = []
        for nfeatures in [500, 1000, 2000]:
            for contrast_threshold in contrast_values:
                for ratio in [0.65, 0.75, 0.85]:
                    candidates.append(
                        AlgorithmCandidate(
                            name="SIFT",
                            params={
                                "nfeatures": nfeatures,
                                "contrastThreshold": contrast_threshold,
                                "edgeThreshold": 10,
                                "matcher": "FLANN",
                                "ratio": ratio,
                                "ransacReprojThreshold": 5.0,
                            },
                        )
                    )
        return candidates

    def _surf_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        texture = min(profile.object_texture_score, profile.scene_texture_score)
        hessian_values = [300, 500, 800] if texture > 200 else [150, 300, 500]

        candidates = []
        for hessian_threshold in hessian_values:
            for ratio in [0.65, 0.75, 0.85]:
                candidates.append(
                    AlgorithmCandidate(
                        name="SURF",
                        params={
                            "hessianThreshold": hessian_threshold,
                            "nOctaves": 4,
                            "nOctaveLayers": 3,
                            "extended": False,
                            "upright": False,
                            "matcher": "FLANN",
                            "ratio": ratio,
                            "ransacReprojThreshold": 5.0,
                        },
                    )
                )
        return candidates

    def _orb_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        edge_density = min(profile.object_edge_density, profile.scene_edge_density)
        scale_values = [1.2, 1.4] if edge_density > 0.04 else [1.1, 1.2]

        candidates = []
        for nfeatures in [500, 1000, 2000]:
            for scale_factor in scale_values:
                for ratio in [0.7, 0.8, 0.9]:
                    candidates.append(
                        AlgorithmCandidate(
                            name="ORB",
                            params={
                                "nfeatures": nfeatures,
                                "scaleFactor": scale_factor,
                                "nlevels": 8,
                                "matcher": "BF",
                                "ratio": ratio,
                                "ransacReprojThreshold": 5.0,
                            },
                        )
                    )
        return candidates

    def _fast_brisk_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        edge_density = min(profile.object_edge_density, profile.scene_edge_density)
        threshold_values = [10, 20, 35] if edge_density < 0.05 else [20, 35, 50]

        candidates = []
        for nfeatures in [800, 1500, 2500]:
            for fast_threshold in threshold_values:
                for ratio in [0.7, 0.8, 0.88]:
                    candidates.append(
                        AlgorithmCandidate(
                            name="FAST_BRISK",
                            params={
                                "nfeatures": nfeatures,
                                "fastThreshold": fast_threshold,
                                "nonmaxSuppression": True,
                                "matcher": "BF",
                                "ratio": ratio,
                                "ransacReprojThreshold": 5.0,
                            },
                        )
                    )
        return candidates

    def _akaze_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        candidates = []
        for threshold in [0.0005, 0.001, 0.003]:
            for ratio in [0.7, 0.8, 0.9]:
                candidates.append(
                    AlgorithmCandidate(
                        name="AKAZE",
                        params={
                            "threshold": threshold,
                            "nOctaves": 4,
                            "matcher": "BF",
                            "ratio": ratio,
                            "ransacReprojThreshold": 5.0,
                        },
                    )
                )
        return candidates

    def _superpoint_superglue_candidates(self, profile: ImageProfile) -> list[AlgorithmCandidate]:
        candidates = []
        for weights in ["outdoor", "indoor"]:
            for keypoint_threshold in [0.003, 0.005, 0.01]:
                for match_threshold in [0.2, 0.35]:
                    candidates.append(
                        AlgorithmCandidate(
                            name="SUPERPOINT_SUPERGLUE",
                            params={
                                "weights": weights,
                                "keypointThreshold": keypoint_threshold,
                                "matchThreshold": match_threshold,
                                "maxKeypoints": 1024,
                                "nmsRadius": 4,
                                "sinkhornIterations": 20,
                                "ransacReprojThreshold": 5.0,
                            },
                        )
                    )
        return candidates
