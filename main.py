from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from agents.pipeline import AutoParameterPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-select and tune feature matching algorithms."
    )
    parser.add_argument("--object", required=True, help="Path to target object image.")
    parser.add_argument("--scene", required=True, help="Path to scene image.")
    parser.add_argument(
        "--results",
        default="results",
        help="Directory for CSV, visualization and report outputs.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional limit for parameter combinations during quick tests.",
    )
    parser.add_argument(
        "--enable-llm-strategy",
        action="store_true",
        help="Use an LLM strategy agent after coarse search to suggest extra candidates.",
    )
    parser.add_argument(
        "--enable-object-mask",
        action="store_true",
        help="Use a lightweight object mask for target feature extraction and visualization.",
    )
    parser.add_argument(
        "--llm-strategy-model",
        default="deepseek-chat",
        help="Model name for the LLM strategy agent.",
    )
    parser.add_argument(
        "--disable-llm-scoring-policy",
        action="store_true",
        help="Disable LLM scoring-policy calls and use only the local adaptive policy.",
    )
    return parser.parse_args()


def read_image(path: str) -> cv2.Mat:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def main() -> None:
    args = parse_args()
    object_path = Path(args.object)
    scene_path = Path(args.scene)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    object_image = read_image(str(object_path))
    scene_image = read_image(str(scene_path))

    output = AutoParameterPipeline(
        max_combinations=args.max_combinations,
        enable_object_mask=args.enable_object_mask,
        enable_llm_strategy=args.enable_llm_strategy,
        llm_strategy_model=args.llm_strategy_model,
        enable_llm_scoring_policy=not args.disable_llm_scoring_policy,
    ).run_and_save(
        object_image=object_image,
        scene_image=scene_image,
        object_path=object_path,
        scene_path=scene_path,
        results_dir=results_dir,
    )
    best_result = output.best_result

    if best_result is not None:
        print("Best algorithm:", best_result.algorithm)
        print("Best params:", best_result.params)
        print("Best score:", f"{best_result.score:.4f}")
        print("Outputs saved to:", results_dir.resolve())
    else:
        print("No valid matching result was produced.")


if __name__ == "__main__":
    main()
