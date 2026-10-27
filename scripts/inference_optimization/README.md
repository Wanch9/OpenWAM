# 推理优化：启动、测速与回退

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
