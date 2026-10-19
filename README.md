# Auto Parameter Agent

一个用于计算机视觉大作业的自动调参原型。系统会自动尝试 SIFT、SURF、ORB、AKAZE、FAST+BRISK，以及可选的 SuperPoint/SuperGlue 特征匹配算法，通过 RANSAC 筛选内点，并根据匹配数量、内点率、重投影误差、投影框合理性、覆盖度和耗时进行综合评价。

项目核心思想是构建“执行-评价-改进”的闭环 Agent：先生成候选算法和参数，运行匹配代码，再用评价函数打分；如果结果仍有改进空间，就继续围绕高分结果调整参数或调用 LLM Agent 给出策略建议。

## 快速开始

准备两张图片：

- `data/object.jpg`：目标物体图
- `data/scene.jpg`：场景图

运行：

```bash
python main.py --object data/object.jpg --scene data/scene.jpg
```

结果会输出到 `results/`：

- `all_results.csv`：所有算法和参数组合的指标
- `best_match.jpg`：最优匹配可视化结果
- `report.md`：自动生成的实验报告
- `decision_log.json`：Agent 决策日志

## 当前支持

- SIFT + BF/FLANN + Lowe ratio test + RANSAC
- SURF + FLANN + Lowe ratio test + RANSAC，需要 `opencv-contrib-python` 的 nonfree 模块
- ORB + BF + Lowe ratio test + RANSAC
- AKAZE + BF + Lowe ratio test + RANSAC
- FAST + BRISK + BF + Lowe ratio test + RANSAC
- SuperPoint + SuperGlue + RANSAC，可选深度学习后端

## 自动调参流程

系统主要由以下 Agent 协作完成：

1. `ImageAnalyzerAgent` 分析目标图和场景图的尺寸、边缘密度和纹理强度。
2. `AlgorithmSelectorAgent` 根据图像特征生成初始算法候选和参数组合。
3. `TuningAgent` 先粗搜，再围绕 Top 结果细搜，并在提升不足时早停。
4. `LLMStrategyAgent` 可选读取 Top/失败结果，建议下一轮参数更新，例如收紧 `ratio`、调整 `ransacReprojThreshold` 或增加特征数量。
5. `ScoringPolicyAgent` 根据图像特征、任务目标和初步匹配结果生成自适应评分权重。
6. `EvaluatorAgent` 使用确定性的评价函数计算最终分数并排序。
7. `DeepPolicyAgent` 判断是否需要补充运行 SuperPoint/SuperGlue。
8. `ReporterAgent` 输出 CSV、报告、可视化图和决策日志。

## 综合评分

当前评分已经不是早期的四项固定公式，而是由多个指标共同决定：

```text
score =
    w1 * inlier_ratio
  + w2 * normalized_inliers
  + w3 * normalized_matches
  + w4 * reprojection_score
  + w5 * homography_quality
  + w6 * coverage_score
  + w7 * speed_score
  + w8 * strictness_score
  - penalties
```

其中：

- `inlier_ratio`：RANSAC 内点率
- `normalized_inliers`：归一化后的内点数量
- `normalized_matches`：归一化后的优质匹配数量
- `reprojection_score`：由重投影误差转换得到，误差越低得分越高
- `homography_quality`：投影框面积、凸性、是否落在场景图内等几何合理性
- `coverage_score`：内点在目标图上的空间覆盖度
- `evidence_score`：证据充分性，综合内点数量、匹配数量、内点率、覆盖度和分布质量
- `spatial_distribution_score`：点分布均匀性，结合 3x3 网格占用和近似共线检测
- `speed_score`：运行速度得分
- `strictness_score`：参数严格性，避免过宽松的 ratio 或 RANSAC 阈值

系统还会对明显不可靠的结果扣分，例如：

- 内点少于 4
- 无法估计 Homography
- 重投影误差过大
- 投影框明显不合理
- 匹配覆盖度过低
- 证据充分性不足
- 内点分布过于集中或近似共线
- 内点率过低
- 参数过于宽松

系统还会输出 `confidence_level`，将结果分为 `high`、`medium`、`low` 和 `unreliable`。置信等级不是替代综合分，而是帮助解释综合分是否有足够证据支撑。例如少量内点即使重投影误差很低，也可能因为证据不足或点分布过于集中而被标记为低置信。

## 自适应评分策略 Agent

不同图像适合的评价重点不同。纹理丰富的包装盒、书本、logo 通常应更强调内点率、重投影误差和几何稳定性；白色杯子、玻璃杯、金属物体等低纹理目标则不应过度依赖匹配数量，而应更重视覆盖度、投影框合理性和低置信度提示。

因此项目新增了 `ScoringPolicyAgent`。它不会直接替代评分函数，而是负责生成评分权重；最终得分仍由 `EvaluatorAgent` 按确定性公式计算。

流程如下：

```text
图像特征 + 任务目标 + 初步匹配结果
        ↓
ScoringPolicyAgent 生成指标权重
        ↓
EvaluatorAgent 使用这些权重重新评分
        ↓
选择最终最优结果
```

如果配置了 DeepSeek API Key，`ScoringPolicyAgent` 会调用 LLM 生成权重和解释；如果没有配置或调用失败，则自动使用本地规则兜底。这样既能利用大模型的策略推理能力，又保留评分结果的稳定性和可复现性。

## DeepSeek 配置

不要把真实 API Key 写进 `config.example.json`。推荐在项目根目录新建 `config.local.json`：

```json
{
  "DEEPSEEK_API_KEY": "你的 DeepSeek API Key",
  "DEEPSEEK_MODEL": "deepseek-chat"
}
```

也可以使用 `.env`：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-chat
```

代码会按以下顺序读取：

1. 系统环境变量
2. `.env`
3. `config.local.json`
4. `config.json`

`.env` 和 `config.local.json` 已加入 `.gitignore`，避免 API Key 被误提交。

## 页面演示

运行 Streamlit 页面：

```bash
streamlit run app.py
```

页面支持：

- 上传目标物体图和场景图
- 选择搜索规模
- 选择评价目标：综合平衡、时间最快、精度最高、数量最优
- 启用目标图自动裁剪
- 启用轻量目标 mask
- 启用 SuperPoint/SuperGlue
- 启用 LLM 策略 Agent
- 启用 LLM 评分策略 Agent
- 启用 LLM 解释 Agent

页面会展示最优算法、参数、匹配可视化、评分拆解、调参轨迹、Top 结果表格、决策日志和 LLM 总结。

## 轻量目标 Mask

对于马克杯、瓶子、鼠标等不规则目标，单纯裁剪成矩形仍可能保留一部分背景。项目提供了一个轻量目标 mask 选项，用于在不引入大型分割模型的情况下减少背景干扰。

启用后，系统会：

1. 根据目标图边界颜色估计背景颜色。
2. 结合颜色距离和边缘信息生成二值 mask。
3. 检查 mask 面积和外接区域是否合理。
4. 只在 mask 内提取目标图特征点。
5. 最终可视化时优先投影 mask 外接区域，而不是整张目标图矩形。

如果 mask 自动生成质量不足，系统会回退到原始整图匹配流程。该功能主要用于减少目标图背景对匹配和画框的影响，但不会凭空增加低纹理物体的有效特征点，因此对白色杯子、玻璃杯等目标仍需要结合匹配质量和框可靠性判断。

## 可选深度学习后端

SuperPoint/SuperGlue 已接入主流程，算法名为 `SUPERPOINT_SUPERGLUE`。如果未配置 PyTorch 或本地模型仓库，对应候选会失败并被自动跳过，不影响传统算法结果。

安装方式：

```bash
pip install -r requirements-deep.txt
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git third_party/SuperGluePretrainedNetwork
```

也可以通过环境变量指定仓库位置：

```bash
set SUPERGLUE_REPO=D:\path\to\SuperGluePretrainedNetwork
```

默认策略是“按需启用”：系统会先运行传统算法，只有当传统最优结果的综合得分、内点率、覆盖度、框合理性或重投影误差不满足阈值时，才运行少量 SuperPoint/SuperGlue 候选。
