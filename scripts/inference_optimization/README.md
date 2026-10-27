# 推理优化：启动、测速与回退

## 按方法逐项叠加（推荐贡献/复现方式）

所有实现位于同一分支，使用配置开关选择方法，不需要按历史提交顺序 cherry-pick。独立开关的定义集中在 `ablation_methods.json`；基线为 `configs/inference_ablation/base.yaml`。这样同一份代码、checkpoint、输入和 seed 可以比较任意合法组合。

先取代码（PR 未合并时）：

```bash
git clone --branch codex/inference-100ms-summary https://github.com/Wanch9/OpenWAM.git
cd OpenWAM
```

只生成配置和实验计划，**不加载模型、不测速**：

```bash
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/plan-001 --prepare-only
```

`plan.json` 为每一阶段保存启用的方法、完整 YAML 路径和可执行命令。默认累计顺序：

| 阶段 | 相对上一阶段增加的唯一方法 |
|---|---|
| 00 | eager 基线：固定热 prompt cache、关闭视频解码，其余新增优化关闭 |
| 01 | compile / CUDA Graph，reduce-overhead |
| 02 | 条件 K/V 缓存 |
| 03 | 条件 padding 裁剪 |
| 04 | 条件 embedding 复用 |
| 05 | 等价 attention 拆分 |
| 06 | 固定形状 action pinning |
| 07 | 跳过重复 eval |
| 08 | timestep 批量准备 |
| 09 | 并行图像处理 |
| 10 | 首帧 prefix prefill 缓存，仍完整计算 10F |
| 11 | 3F residual 近似复用（第一次引入残差近似） |
| 12 | time-MLP 缓存 |
| 13 | RoPE 网格缓存 |
| 14 | lightweight residual 路径 |

实际运行整套实验：去掉 `--prepare-only` 并使用新的输出目录。每阶段独立进程加载模型，记录加载/预热，不让上阶段的可变缓存留在下阶段；会多次加载大模型，运行时间显著长于单组测速。脚本拒绝覆盖已有目录，不自动重试失败阶段。生成后的 YAML 也可直接交给 `scripts/deploy.py --config ...`。

只测一个方法：

```bash
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/compile-only --methods compile
```

任意组合（包含 eager 基线及指定组合两组）：

```bash
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/kv-compact --methods compile conditional_kv compact_context
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/velocity --methods compile velocity memoize_similarity
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/autotune --methods compile max_autotune
python scripts/inference_optimization/run_ablation.py --checkpoint /path/to/checkpoint --output-dir results/rope-kernel --methods compile triton_rope
```

每次都可加 `--prepare-only`。`compact_context/context_embedding` 依赖 `conditional_kv`；`time_cache` 依赖 `batch_timesteps`；`lightweight` 依赖 `residual3f`；`memoize_similarity` 依赖 `velocity`。依赖必须显式列出，脚本不会偷偷添加其他方法。`velocity` 与 `residual3f`、两种 Triton backend 分别互斥。固定 prompt cache/不解码视频是基线条件，不在这个 ladder 中做单独消融。已撤销的 QKV 融合没有实现开关。

默认仍是 LIBERO EEF10 合成输入；真实数据传 `--fixtures`，非 LIBERO 另传 `--action-contract generic`。`--no-prefix-verify` 仅在验证观测不变量后使用；默认保留检查，因此延迟不保证复现历史关闭检查的数字。

完成后生成 `summary.json` / `summary.md`：每阶段均值、P95、相对基线倍数、相对**同一个 eager 基线**的完整动作误差。各阶段 JSON/NPZ 另保留逐请求统计与全部动作。自带 benchmark 的阶段内 full10F 参考保留作局部对照，不替代公共基线。累计相邻差值是该上下文下的增量收益，不代表方法在所有组合中的独立收益。

新增 ladder 和移植代码按用户要求未运行验证。以上是可执行的复现入口，不是宣称这些新组合已经实测通过。

实现从已测快照 `4804e061aa5a4d53601ce5bb80e5988035509575` 提取。包含 3F 残差复用、观测首帧缓存、固定调度时间缓存、RoPE 缓存、轻量复用路径及前期条件 K/V、注意力和前处理优化。训练优化与机器人通信协议不在本次移植中。

## 环境与资产

使用项目的依赖安装流程；历史 CUDA 测量使用 Linux、PyTorch 2.7.1+cu128、Triton 3.3.1、RTX 6000D/4090。当前仓库通用依赖范围可能与历史环境不同，换环境需要重测。`rope_kernel: torch` 是默认；随附 Triton 算子仅是实验回退完整性所需组件，不宣称完整请求加速。

需要自包含 checkpoint 目录：`config.yaml`、模型 safetensors、normalization 与模型加载器所需 tokenizer/VAE/T5 资产；模型和数据不随代码上传。预留整个模型及编译缓存所需显存。本 PR 没有移植 4090 部署的跨卡文本编码器放置；单卡不能仅按报告中的 14.28 GiB DiT 主卡占用估算完整模型需求。G1D 等自定义数据集还需对应的 checkpoint 适配器。

## 启动

从仓库根目录执行，将示例 checkpoint 路径替换为实际路径：

```bash
python scripts/deploy.py --config configs/deploy_realtimewam_3f_v2.yaml --ckpt-dir /path/to/checkpoint --device cuda:0
```

默认只绑定 127.0.0.1:8849。`deploy_realtimewam_3f.yaml` 是原 3F 对照，`deploy_realtimewam_4f.yaml` 为 4F 候选。完整 10F 回退：

```bash
python scripts/deploy.py --config configs/deploy_realtimewam_10f_dense.yaml --ckpt-dir /path/to/checkpoint --device cuda:0
```

适用 Wan TI2V、joint self-attention、mutual + first-frame-causal、batch=1、同步 10 步、CFG=1、eval/no-grad。3F 配置关闭旧速度缓存，两种近似复用互斥。配置中的 prefix invariant 校验关闭是历史实测选择，不表示可更改请求内观测/条件/权重；下面的测速默认开启该校验。

## 完整请求配对测速

```bash
python scripts/inference_optimization/benchmark_realtimewam.py --checkpoint /path/to/checkpoint --config configs/deploy_realtimewam_3f_v2.yaml --output results/3f-run-001.json --phases full10F prefix_residual3F time_rope_light_prefix_residual3F --warmups 6 --samples 60 --seeds 42 123
```

默认使用确定性合成 LIBERO EEF10 输入，只能搭配相应模型契约。真实输入使用 `--fixtures fixtures.json`。支持的 JSON 示例（相机键、state 维度和单位必须对应 checkpoint）：

```json
[{"id":"case0","observation":{"images":{"head_camera":{"path":"head.jpg"},"left_wrist_camera":{"path":"wrist.jpg"},"right_wrist_camera":null},"prompt":"pick up the object","state":[0,0,0,1,0,0,0,1,0,0]}}]
```

图片路径相对 fixtures 文件解析，也可使用 base64 字符串。对于 EEF20 等非 LIBERO 契约，添加 `--action-contract generic --fixtures fixtures.json`，仅报告通用归一化/原动作单位误差，不套用 LIBERO 的 OSC 单位；该路径要求原动作维数等于有效统一槽位数量。脚本当前固定验证 32 步、80 维统一动作输出。

默认相对完整 `full10F` 比较；若仅检查第二轮是否保持原 3F 结果，可指定 `--reference-phase prefix_residual3F`，此时不能解释为相对 10F 的质量误差。验证不变量后，`--no-prefix-verify` 对齐历史快速配置。`--no-compile` 用于 eager 排查；`--profile` 单独增加未计时的 profiler 请求。

每次计时都重置动作队列、同步 CUDA，并确认生成完整的新动作段。记录模型加载/预热、各请求延迟、均值/P95、完整动作 NPZ、配对误差、实际 forward/cache 计数与编译状态。结果文件/数组目录已存在时拒绝覆盖。完整计时包含服务端预处理/VAE/条件准备/采样/后处理，不包含网络、编译、预热与 profiler。

LIBERO EEF10 的进一步动作偏差分析：

```bash
python scripts/inference_optimization/analyze_realtimewam.py results/3f-run-001.json --output results/3f-analysis-001.json
```

分析器只适用于 LIBERO EEF10；泛用动作契约请使用 benchmark 内的通用误差及 NPZ。任何离线动作偏差都不等于真机成功率。

## 验证

本次移植的本地测试未完成：首次收集缺少环境依赖；补齐依赖后的运行按用户要求停止，此后不再运行测试或测速。没有新的 CUDA 基准结果，不将历史实测视为本次移植已验证。

核心回归文件：`tests/test_{conditional_kv,residual_cache,prefix_cache,time_cache,rope_cache,lightweight_reuse,realtime_engine_guards,static_serving,parallel_images,fused_norm_rope}.py`。CPU 可验证缓存生命周期、失效、异常和数值回退；CUDA/编译/图重放测试需要 GPU。移植整理的测试结果见 PR，历史 GPU 延迟见 [实测总结](../../assets/openwam_usage_docs/inference-100ms-summary.zh.md)。
