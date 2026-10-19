from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from agents.llm_explainer import LLMExplanationAgent
from agents.pipeline import AutoParameterPipeline
from agents.reporter import ReporterAgent
from agents.settings import get_deepseek_api_key, get_deepseek_model


SCORE_LABELS = {
    "inlier_ratio": "内点率",
    "inlier_count": "内点数量",
    "match_count": "匹配数量",
    "reprojection": "重投影误差",
    "geometry": "框合理性",
    "coverage": "覆盖度",
    "evidence": "证据充分性",
    "spatial_distribution": "点分布均匀性",
    "speed": "速度",
    "strictness": "参数严格度",
}

ALGORITHM_NOTES = [
    {
        "算法": "SIFT",
        "类型": "传统局部特征",
        "说明": "尺度和旋转鲁棒性强，适合纹理、文字、LOGO 丰富的目标，速度相对较慢。",
    },
    {
        "算法": "SURF",
        "类型": "传统局部特征",
        "说明": "SIFT 的加速思想，依赖 opencv-contrib nonfree 模块；环境不支持时会自动跳过。",
    },
    {
        "算法": "ORB",
        "类型": "传统二进制特征",
        "说明": "速度快，适合轻量场景，但大透视和复杂背景下稳定性通常弱于 SIFT。",
    },
    {
        "算法": "AKAZE",
        "类型": "传统二进制特征",
        "说明": "利用非线性尺度空间，适合边缘和纹理明显的目标，速度和鲁棒性较均衡。",
    },
    {
        "算法": "FAST+BRISK",
        "类型": "传统角点+描述子",
        "说明": "FAST 负责快速检测角点，BRISK 负责生成描述子，适合比较快的候选搜索。",
    },
    {
        "算法": "SuperPoint/SuperGlue",
        "类型": "深度学习匹配",
        "说明": "已接入主流程为 SUPERPOINT_SUPERGLUE；需要 PyTorch 和本地 SuperGlue 预训练仓库，未配置时会自动跳过。",
    },
]


def read_uploaded_image(uploaded_file) -> cv2.Mat | None:
    if uploaded_file is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as file:
        file.write(uploaded_file.getbuffer())
        temp_path = file.name

    image = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    Path(temp_path).unlink(missing_ok=True)
    return image


def bgr_to_rgb(image: cv2.Mat) -> cv2.Mat:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def result_rows(results):
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "排名": rank,
                "阶段": result.search_phase,
                "步骤": result.search_step,
                "算法": result.algorithm,
                "综合得分": round(result.score, 4),
                "优质匹配": result.num_good_matches,
                "RANSAC内点": result.num_inliers,
                "内点率": round(result.inlier_ratio, 4),
                "重投影误差": round(result.reprojection_error, 3)
                if result.reprojection_error is not None
                else None,
                "覆盖度": round(result.match_coverage, 4),
                "框合理性": round(result.homography_quality, 4),
                "证据充分性": round(result.evidence_score, 4),
                "点分布均匀性": round(result.spatial_distribution_score, 4),
                "置信等级": result.confidence_level,
                "框类型": box_type_label(result.visualization_box_type),
                "定位状态": result.localization_status,
                "耗时ms": round(result.runtime_ms, 2),
                "参数": result.params,
            }
        )
    return rows


def failed_result_rows(results):
    rows = []
    for result in results:
        rows.append(
            {
                "阶段": result.search_phase,
                "步骤": result.search_step,
                "算法": result.algorithm,
                "错误/跳过原因": result.error,
                "参数": result.params,
            }
        )
    return rows


def score_breakdown_rows(result):
    return [
        {
            "指标": SCORE_LABELS.get(name, name),
            "归一化得分": round(value, 4),
        }
        for name, value in result.score_breakdown.items()
    ]


def search_trace_rows(results):
    ordered = sorted(results, key=lambda item: item.search_step)
    best_so_far = 0.0
    rows = []
    for result in ordered:
        best_so_far = max(best_so_far, result.score)
        rows.append(
            {
                "步骤": result.search_step,
                "阶段": result.search_phase,
                "算法": result.algorithm,
                "当前得分": result.score,
                "历史最好": best_so_far,
            }
        )
    return rows


def phase_summary_rows(results):
    rows = []
    for phase, group in pd.DataFrame(result_rows(results)).groupby("阶段"):
        rows.append(
            {
                "阶段": phase,
                "尝试次数": len(group),
                "最高得分": round(group["综合得分"].max(), 4),
                "平均耗时ms": round(group["耗时ms"].mean(), 2),
            }
        )
    return rows


def classify_failure_reasons(result):
    reasons = []
    error_text = str(result.error or "")
    lower_error = error_text.lower()
    if result.error:
        if (
            "super" in result.algorithm.lower()
            or "superglue" in lower_error
            or "superpoint" in lower_error
            or "torch" in lower_error
            or "pytorch" in lower_error
            or "weight" in lower_error
            or "repo" in lower_error
        ):
            reasons.append("SuperPoint/SuperGlue 环境不可用")
        else:
            reasons.append("运行失败或候选跳过")
    if result.num_good_matches < 8:
        reasons.append("匹配数量不足")
    if result.num_inliers < 4:
        reasons.append("内点不足")
    if result.error is None and result.homography is None:
        reasons.append("Homography 失败")
    if result.error is None and result.match_coverage < 0.15:
        reasons.append("覆盖度低")
    if result.error is None and result.confidence_level == "unreliable":
        reasons.append("低置信结果")
    return reasons or ["其他原因"]


def failure_reason_summary_rows(valid_results, failed_results):
    reason_counts = {}
    detail_rows = []
    for result in list(valid_results) + list(failed_results):
        for reason in classify_failure_reasons(result):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            detail_rows.append(
                {
                    "算法": result.algorithm,
                    "阶段": result.search_phase,
                    "步骤": result.search_step,
                    "原因": reason,
                    "综合得分": round(result.score, 4),
                    "优质匹配": result.num_good_matches,
                    "RANSAC内点": result.num_inliers,
                    "覆盖度": round(result.match_coverage, 4),
                    "错误信息": result.error or "",
                }
            )
    summary_rows = [
        {"失败/风险原因": reason, "次数": count}
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return summary_rows, detail_rows


def algorithm_metric_rows(results):
    best_by_algorithm = {}
    for result in results:
        current = best_by_algorithm.get(result.algorithm)
        if current is None or result.score > current.score:
            best_by_algorithm[result.algorithm] = result

    rows = []
    for algorithm, result in sorted(best_by_algorithm.items()):
        rows.append(
            {
                "算法": algorithm,
                "综合得分": round(result.score, 4),
                "内点率": round(result.inlier_ratio, 4),
                "匹配数量得分": round(result.score_breakdown.get("match_count", 0.0), 4),
                "覆盖度": round(result.match_coverage, 4),
                "几何质量": round(result.homography_quality, 4),
                "速度得分": round(result.score_breakdown.get("speed", 0.0), 4),
                "优质匹配数": result.num_good_matches,
                "RANSAC内点数": result.num_inliers,
                "耗时ms": round(result.runtime_ms, 2),
            }
        )
    return rows


def best_explanation(best) -> str:
    reprojection = (
        f"{best.reprojection_error:.3f}"
        if best.reprojection_error is not None
        else "无有效单应性"
    )


def objective_explanation(objective: str) -> str:
    if objective == "时间最快":
        return "当前目标偏向运行速度，会提高耗时指标权重，但仍保留基本几何质量约束。"
    if objective == "精度最高":
        return "当前目标偏向匹配精度，会提高内点率、重投影误差、框合理性和覆盖度权重。"
    if objective == "数量最优":
        return "当前目标偏向匹配数量，会提高优质匹配数和 RANSAC 内点数权重。"
    return "当前目标为综合平衡，同时考虑时间、精度、数量和几何稳定性。"
    return (
        f"本次 Agent 经过粗搜和细搜后选择 **{best.algorithm}**。"
        f"它的 RANSAC 内点数为 **{best.num_inliers}**，内点率为 **{best.inlier_ratio:.4f}**，"
        f"重投影误差为 **{reprojection}**，投影框合理性得分为 **{best.homography_quality:.4f}**，"
        f"内点覆盖度为 **{best.match_coverage:.4f}**。"
        "综合评分不只追求匹配数量，也会惩罚几何不稳定、覆盖范围过小和耗时过高的结果。"
    )


def reliability_warning(best) -> str | None:
    if best.match_coverage < 0.15:
        return (
            "当前最优结果的内点覆盖度过低，匹配点主要集中在目标的一小块区域。"
            "这类单应性会把局部匹配外推到整张目标图，绿色框可能明显偏大或偏斜。"
            "建议上传只包含目标物体本体的裁剪图，或换用背景更干净的目标图。"
        )
    if best.match_coverage < 0.30:
        return (
            "当前结果的内点覆盖度偏低，定位到了目标，但外框可靠性一般。"
            "如果需要更贴边的框，请尽量裁掉目标图中的桌面、背景和无关区域。"
        )
    return None


def mask_ratio_status(mask_area_ratio: float) -> str:
    if mask_area_ratio < 0.05:
        return "偏小，可能漏掉目标区域"
    if mask_area_ratio > 0.85:
        return "偏大，可能没有有效去除背景"
    if mask_area_ratio > 0.65:
        return "略大，适合目标本身占图较多的情况"
    return "合理"


def format_progress_event(event: dict) -> str:
    agent = event.get("agent", "Agent")
    action = event.get("action", "处理中")
    phase = event.get("phase")
    evaluated = event.get("evaluated_count")
    total = event.get("total_count")
    best_algorithm = event.get("best_algorithm")
    best_score = event.get("best_score")
    algorithm = event.get("algorithm")

    parts = [f"**{agent}**: {action}"]
    if phase:
        parts.append(f"阶段 `{phase}`")
    if algorithm:
        parts.append(f"当前算法 `{algorithm}`")
    if evaluated is not None:
        progress_text = f"已评估 `{evaluated}`"
        if total is not None:
            progress_text += f" / `{total}`"
        parts.append(progress_text)
    if best_algorithm is not None:
        parts.append(f"当前最佳 `{best_algorithm}`")
    if best_score is not None:
        parts.append(f"得分 `{best_score}`")
    return " | ".join(parts)


def box_type_label(box_type: str) -> str:
    labels = {
        "homography": "Homography 投影框",
        "homography_mask": "Mask-aware Homography 投影框",
        "stabilized": "稳定定位框",
        "low_confidence": "低置信局部框",
        "not_drawn": "未绘制定位框",
    }
    return labels.get(box_type, box_type or "未知")


def decision_summary_rows(output, best):
    scoring_source = "未使用"
    scoring_reason = ""
    if output.scoring_policy is not None:
        scoring_source = "LLM" if output.scoring_policy.used_llm else "本地规则"
        scoring_reason = output.scoring_policy.reason

    strategy_state = "未启用"
    if output.strategy_plan is not None:
        strategy_state = (
            f"{'LLM' if output.strategy_plan.used_llm else '本地规则'}，"
            f"{len(output.strategy_plan.suggestions)} 条建议"
        )

    return [
        {
            "项目": "最终算法",
            "结果": best.algorithm,
            "说明": f"综合得分 {best.score:.4f}，置信等级 {best.confidence_level}",
        },
        {
            "项目": "框类型",
            "结果": box_type_label(best.visualization_box_type),
            "说明": best.visualization_box_reason,
        },
        {
            "项目": "目标 mask",
            "结果": "启用" if output.object_mask_box is not None else "未启用/未通过质量检查",
            "说明": (
                f"面积占比 {output.object_mask_area_ratio:.4f}，区域 {output.object_mask_box}"
                if output.object_mask_area_ratio is not None
                else "使用整张目标图参与匹配"
            ),
        },
        {
            "项目": "评分策略",
            "结果": scoring_source,
            "说明": scoring_reason,
        },
        {
            "项目": "调参策略",
            "结果": strategy_state,
            "说明": "粗搜后围绕高分结果继续细搜，并可由 LLM 建议额外候选。",
        },
        {
            "项目": "深度学习策略",
            "结果": "已启用" if output.deep_learning_used else "未启用",
            "说明": output.deep_learning_reason,
        },
    ]


st.set_page_config(
    page_title="Auto Parameter Agent",
    page_icon="CV",
    layout="wide",
)

st.title("基于 Agent 的图像匹配算法自动选择及调参")

with st.sidebar:
    st.header("输入")
    configured_deepseek_api_key = get_deepseek_api_key()
    configured_deepseek_model = get_deepseek_model()
    object_file = st.file_uploader(
        "目标物体图",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    scene_file = st.file_uploader(
        "场景图",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    search_mode = st.selectbox(
        "搜索规模",
        ["完整搜索", "快速演示"],
        index=0,
    )
    objective = st.selectbox(
        "算法选择目标",
        ["综合平衡", "时间最快", "精度最高", "数量最优"],
        index=0,
        help="综合平衡为默认策略；其他选项会改变 EvaluatorAgent 的评分权重。",
    )
    enable_object_crop = st.checkbox(
        "启用目标图自动裁剪",
        value=False,
        help="默认关闭，避免误裁目标。只有目标图背景很多且边界清楚时再启用。",
    )
    enable_object_mask = st.checkbox(
        "启用轻量目标 mask",
        value=False,
        help="基于目标图边界颜色自动估计物体区域，只在 mask 内提取特征，并优先投影 mask 外接框。",
    )
    include_deep_learning = st.checkbox(
        "启用 SuperPoint/SuperGlue",
        value=True,
        help="需要 PyTorch 和 third_party/SuperGluePretrainedNetwork。未配置成功时会在跳过结果中显示原因。",
    )
    deep_learning_policy = st.selectbox(
        "深度学习策略",
        ["按需启用", "总是启用", "从不启用"],
        index=0,
        help="按需启用会先跑传统算法，只有传统结果不够好时再运行 SuperPoint/SuperGlue。",
    )
    use_llm_strategy = st.checkbox(
        "启用 LLM 策略 Agent",
        value=False,
        help="在粗搜后读取 Top/失败结果，让大模型建议下一轮参数更新；程序会校验参数范围后再执行。",
    )
    use_llm_scoring_policy = st.checkbox(
        "启用 LLM 评分策略 Agent",
        value=True,
        help="根据图像特征、任务目标和初步结果自适应生成综合评分权重；无 API Key 时使用本地规则兜底。",
    )
    use_llm_explainer = st.checkbox(
        "启用 LLM 解释 Agent",
        value=False,
        help="用于生成最终实验总结。可使用环境变量 DEEPSEEK_API_KEY，或在下方临时输入。",
    )
    deepseek_api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="可留空，自动读取 .env / config.local.json / DEEPSEEK_API_KEY",
        disabled=not (use_llm_explainer or use_llm_strategy or use_llm_scoring_policy),
    )
    if configured_deepseek_api_key:
        st.caption("已检测到本地 DeepSeek API Key；留空也会自动使用。")
    deepseek_models = ["deepseek-chat", "deepseek-reasoner"]
    deepseek_model = st.selectbox(
        "LLM 模型",
        deepseek_models,
        index=deepseek_models.index(configured_deepseek_model)
        if configured_deepseek_model in deepseek_models
        else 0,
        disabled=not (use_llm_explainer or use_llm_strategy or use_llm_scoring_policy),
    )
    st.caption("算法池：SIFT / SURF / ORB / FAST+BRISK / AKAZE / SuperPoint+SuperGlue")
    run_clicked = st.button("运行自动选择与调参", type="primary")

object_image = read_uploaded_image(object_file)
scene_image = read_uploaded_image(scene_file)

preview_left, preview_right = st.columns(2)
with preview_left:
    st.subheader("目标物体图")
    if object_image is not None:
        st.image(bgr_to_rgb(object_image), use_container_width=True)
with preview_right:
    st.subheader("场景图")
    if scene_image is not None:
        st.image(bgr_to_rgb(scene_image), use_container_width=True)

if run_clicked:
    if object_image is None or scene_image is None:
        st.warning("请先上传目标物体图和场景图。")
        st.stop()

    max_combinations = 18 if search_mode == "快速演示" else None
    results_dir = Path("results")
    effective_deepseek_api_key = deepseek_api_key or configured_deepseek_api_key

    with st.status("Agent 正在分析图像、选择算法并自动调参...", expanded=True) as status:
        progress_line = st.empty()
        progress_log = st.container()

        def on_progress(event: dict) -> None:
            message = format_progress_event(event)
            progress_line.markdown(message)
            with progress_log:
                st.write(message)

        output = AutoParameterPipeline(
            max_combinations=max_combinations,
            include_deep_learning=include_deep_learning,
            deep_learning_policy=deep_learning_policy,
            objective=objective,
            enable_object_crop=enable_object_crop,
            enable_object_mask=enable_object_mask,
            enable_llm_strategy=use_llm_strategy,
            llm_strategy_api_key=effective_deepseek_api_key,
            llm_strategy_model=deepseek_model,
            enable_llm_scoring_policy=use_llm_scoring_policy,
            progress_callback=on_progress,
        ).run(
            object_image=object_image,
            scene_image=scene_image,
        )

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
            object_path=Path(object_file.name),
            scene_path=Path(scene_file.name),
            image_profile=output.image_profile,
            results=output.results,
            best_result=output.best_result,
        )
        llm_summary = None
        if use_llm_explainer:
            status.update(label="LLM 解释 Agent 正在生成总结...", state="running")
            llm_summary = LLMExplanationAgent(
                api_key=effective_deepseek_api_key,
                model=deepseek_model,
            ).generate(
                image_profile=output.image_profile,
                results=output.results,
                failed_results=output.failed_results,
                best_result=output.best_result,
                deep_learning_reason=output.deep_learning_reason,
                decision_log=output.decision_log,
            )
            reporter.write_llm_summary(llm_summary)
        status.update(label="自动选择与调参完成", state="complete")

    if output.best_result is None:
        st.error("没有得到有效匹配结果，请换一组纹理更明显的图片。")
        st.stop()

    best = output.best_result
    st.subheader("最优结果")
    metric_cols = st.columns(6)
    metric_cols[0].metric("算法", best.algorithm)
    metric_cols[1].metric("综合得分", f"{best.score:.4f}")
    metric_cols[2].metric("RANSAC 内点", best.num_inliers)
    metric_cols[3].metric("内点率", f"{best.inlier_ratio:.4f}")
    metric_cols[4].metric(
        "重投影误差",
        f"{best.reprojection_error:.3f}" if best.reprojection_error is not None else "N/A",
    )
    metric_cols[5].metric("耗时", f"{best.runtime_ms:.2f} ms")
    confidence_cols = st.columns(3)
    confidence_cols[0].metric("证据充分性", f"{best.evidence_score:.4f}")
    confidence_cols[1].metric("点分布均匀性", f"{best.spatial_distribution_score:.4f}")
    confidence_cols[2].metric("置信等级", best.confidence_level)
    if best.confidence_reason:
        st.caption(f"置信说明: {best.confidence_reason}")

    box_cols = st.columns(3)
    box_cols[0].metric("框类型", box_type_label(best.visualization_box_type))
    box_cols[1].metric(
        "Mask",
        "启用" if output.object_mask_box is not None else "未启用",
    )
    box_cols[2].metric(
        "Mask 面积占比",
        f"{output.object_mask_area_ratio:.4f}"
        if output.object_mask_area_ratio is not None
        else "N/A",
    )
    if best.visualization_box_reason:
        st.caption(f"框说明: {best.visualization_box_reason}")

    with st.expander("本次 Agent 决策链路总结", expanded=True):
        st.dataframe(
            pd.DataFrame(decision_summary_rows(output, best)),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("查看最优参数", expanded=False):
        st.code(str(best.params), language="python")
    if output.object_crop_box is not None:
        st.caption(f"已自动裁剪目标图背景，裁剪区域: {output.object_crop_box}")
        with st.expander("查看预处理后的目标图", expanded=False):
            st.image(
                bgr_to_rgb(output.object_image_used),
                caption="用于匹配的裁剪后目标图",
                use_container_width=True,
            )
    if output.object_mask_box is not None:
        st.caption(f"已启用目标 mask，目标区域外接框: {output.object_mask_box}")
        if output.object_mask_area_ratio is not None:
            st.caption(
                "mask 面积占比: "
                f"{output.object_mask_area_ratio:.4f} "
                f"({mask_ratio_status(output.object_mask_area_ratio)})"
            )
    warning = reliability_warning(best)
    if warning is not None:
        st.warning(warning)

    match_path = results_dir / "best_match.jpg"
    if match_path.exists():
        st.image(str(match_path), caption="最优匹配可视化", use_container_width=True)

    tab_overview, tab_log, tab_score, tab_visual, tab_trace, tab_table, tab_llm = st.tabs(
        ["Agent 决策", "决策日志", "评分拆解", "可视化分析", "调参过程", "Top 结果", "LLM 总结"]
    )

    with tab_overview:
        st.markdown(best_explanation(best))
        st.info(f"当前算法选择目标：{output.objective}。{objective_explanation(output.objective)}")
        if output.scoring_policy is not None:
            source = "LLM" if output.scoring_policy.used_llm else "本地规则"
            st.info(f"评分策略来源：{source}。{output.scoring_policy.reason}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"指标": SCORE_LABELS.get(name, name), "权重": round(value, 4)}
                        for name, value in output.scoring_policy.weights.items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("当前算法库说明", expanded=False):
            st.dataframe(pd.DataFrame(ALGORITHM_NOTES), use_container_width=True, hide_index=True)
        st.info(output.deep_learning_reason)
        profile_cols = st.columns(4)
        profile_cols[0].metric("目标边缘密度", f"{output.image_profile.object_edge_density:.4f}")
        profile_cols[1].metric("场景边缘密度", f"{output.image_profile.scene_edge_density:.4f}")
        profile_cols[2].metric("目标纹理强度", f"{output.image_profile.object_texture_score:.2f}")
        profile_cols[3].metric("场景纹理强度", f"{output.image_profile.scene_texture_score:.2f}")
        phase_summary = pd.DataFrame(phase_summary_rows(output.results))
        st.dataframe(phase_summary, use_container_width=True, hide_index=True)

    with tab_log:
        st.dataframe(pd.DataFrame(output.decision_log.rows()), use_container_width=True, hide_index=True)
        with st.expander("文本版决策链", expanded=False):
            for index, entry in enumerate(output.decision_log.entries, start=1):
                st.markdown(
                    f"**{index}. {entry.agent}**  \n"
                    f"动作：{entry.action}  \n"
                    f"依据：{entry.reason}  \n"
                    f"细节：`{entry.details}`"
                )

    with tab_score:
        breakdown = pd.DataFrame(score_breakdown_rows(best))
        st.bar_chart(breakdown, x="指标", y="归一化得分", use_container_width=True)
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    with tab_visual:
        st.subheader("失败原因统计")
        reason_summary, reason_details = failure_reason_summary_rows(
            output.results,
            output.failed_results,
        )
        if reason_summary:
            reason_summary_df = pd.DataFrame(reason_summary)
            st.bar_chart(
                reason_summary_df,
                x="失败/风险原因",
                y="次数",
                use_container_width=True,
            )
            st.dataframe(reason_summary_df, use_container_width=True, hide_index=True)
            with st.expander("查看失败/风险明细", expanded=False):
                st.dataframe(
                    pd.DataFrame(reason_details),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.success("当前候选没有明显失败或风险原因。")

        st.subheader("各算法指标对比")
        metric_rows = pd.DataFrame(algorithm_metric_rows(output.results))
        if not metric_rows.empty:
            chart_columns = ["综合得分", "内点率", "匹配数量得分", "覆盖度", "几何质量", "速度得分"]
            st.bar_chart(
                metric_rows.set_index("算法")[chart_columns],
                use_container_width=True,
            )
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)
            st.caption(
                "说明：每个算法取该算法当前最高综合得分的候选结果；匹配数量得分和速度得分来自评分拆解中的归一化指标。"
            )
        else:
            st.info("暂无可用于算法对比的有效结果。")

    with tab_trace:
        trace = pd.DataFrame(search_trace_rows(output.results))
        st.line_chart(trace, x="步骤", y=["当前得分", "历史最好"], use_container_width=True)
        st.dataframe(trace, use_container_width=True, hide_index=True)

    with tab_table:
        rows = pd.DataFrame(result_rows(output.results[:20]))
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if output.failed_results:
            st.subheader("跳过或失败的候选")
            failed_rows = pd.DataFrame(failed_result_rows(output.failed_results[:30]))
            st.dataframe(failed_rows, use_container_width=True, hide_index=True)

    with tab_llm:
        if use_llm_explainer and llm_summary:
            st.markdown(llm_summary)
            st.caption("总结已保存到 results/llm_summary.md")
        else:
            st.info("侧边栏勾选“启用 LLM 解释 Agent”后，会在这里生成最终实验总结。")
else:
    st.info("上传两张图片后，点击左侧按钮运行自动选择与调参。")
