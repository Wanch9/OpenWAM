# OpenWAM 约 100 ms 推理优化总结

整理日期：2026-10-27。本文记录已做的推理优化，不涉及重新训练或冻结 Wan 的训练实验。

**“100 ms”指特定条件下约 100 ms 的新动作段生成延迟，不是所有设备上的保证。** 后续部署记录引用的 RTX 6000D / LIBERO 结果为 **101.84 ms**；本次整理直接复核到逐次结果的 RTX 4090 / G1D 实验为 **112.50 ms 均值、114.93 ms P95**。

本文是文档与实测摘要交付，不包含实现代码的合并。下方模块名、配置名指已测实现快照 `4804e061aa5a4d53601ce5bb80e5988035509575`，不能假定当前仓库 main 已包含这些实现；仅复制配置不能启用优化。

## 1. 核心：10 次更新，3 次完整 Transformer

原完整路径每个去噪更新都计算联合视频/动作 Transformer。3F 保留原来的 **10 次 Euler 更新、32 步动作输出、BF16、384×320 输入**，仅在零起始索引 `[0, 5, 9]` 更新完整 Transformer。

在完整计算步，分别保存视频流和动作流的输入到输出残差：

```text
r = Transformer(h_refresh) - h_refresh
```

其余 7 步用当前输入与最近一次残差构造近似隐状态：

```text
h_out ≈ h_current + r_cached
```

**每一步仍计算当前视频/动作输出头并执行 Euler 更新。** 这里的 3F 是 3 次完整 Transformer 求值，不是 3 帧视频，也不是把采样器改为 3 步。

这是主要的近似计算策略；不是与完整 10F 数学等价的加速。它也不是旧的 velocity cache：旧策略直接复用速度预测、跳过整个前向及输出头。3F 配置关闭旧 `dit_cache`，两种复用不叠加。

实现位置（已测快照）：`openwam/deploy/optimizations/residual_cache.py`。

## 2. 首帧只计算一次

在 `joint_self_attn`、`mutual + first_frame_causal` 条件下，干净观测首帧不读取未来视频和动作。一次生成过程中，首帧、条件与权重不变，因此可以：

1. prefill 一次首帧，保存各层 K/V 与最终首帧状态；
2. 后续完整 Transformer 求值只更新未来视频和动作 tokens；
3. 这些 tokens 继续读取缓存的首帧 K/V。

这不是冻结整个 Wan，也不是跨不同观测直接复用首帧。缓存限定在单次生成内，正常结束及异常退出均清空。实现目前限制 Wan TI2V、batch=1、规定的 mask 和无梯度推理；条件改变不能继续命中。

实现位置：`openwam/deploy/optimizations/prefix_cache.py`。已测配置关闭逐次 invariant 校验，依赖固定输入契约；迁移到新的输入/架构时应重新验证。

## 3. 第二轮：消除仍然重复的准备工作

| 手段 | 去掉的工作 | 边界 |
|---|---|---|
| 固定调度 time cache | 同一调度的时间 embedding、调制 MLP 重算 | 缓存依赖权重、形状、dtype/device、执行模式和调度；条件失效要重算 |
| RoPE cache | 固定 token 网格的位置频率重复构造 | 网格、设备、精度等变化不能复用旧值 |
| 轻量 residual reuse | 7 个复用步中只有 Transformer 需要的时间调制、context、RoPE 准备 | 当前输入 embedding 和两个当前输出头仍运行 |
| 条件 K/V 缓存 | 每层重复投影请求内不变的文本/本体条件 | 不缓存变化的视频/动作自注意力 K/V |
| 文本 padding 裁剪 | 对无效 padding 的注意力计算 | 保留有效 proprio token 与 mask，按桶处理形状 |
| 首帧注意力拆分 | 通用 mask 带来的执行成本 | 只在原可见性可等价拆分时启用 |
| `torch.compile(reduce-overhead)` / CUDA Graph | Python 调度和小 kernel 提交开销 | 首次编译和 shape 预热单独计时 |
| CPU 与小算子整理 | 重复 `.eval()`、小 timestep 传输、动态索引同步、串行相机解码 | 固定形状 `where`、批量传输、两个图像 worker；保留回退 |

这些准备和执行优化旨在保持既定 3F 算法结果，不消除 3F 本身相对 10F 的近似误差。

实现位置：`openwam/deploy/optimizations/{time_cache,rope_cache,lightweight_reuse,conditional_kv}.py`（具体条件 K/V 文件名以实现快照为准）、部署 engine、视频/动作 backbone 和服务端预处理。

## 4. 实测结果与计时范围

| 实验 | 环境/任务 | 完整请求均值 | P95 | 证据范围 |
|---|---|---:|---:|---|
| 前期 v3 | RTX 6000D，合成观测 | 166.93 ms | 168.53 ms | 120 次；旧速度缓存下 4 次实际前向、6 次速度复用 |
| 第二轮 3F | RTX 6000D / LIBERO | 101.84 ms | 未在本次材料中复核 | 后续部署报告引用；本次未取得该轮逐次记录 |
| 原 3F | RTX 4090 / G1D | 126.02 ms | 129.11 ms | 同轮 24 次热态新动作段请求 |
| 第二轮 3F | RTX 4090 / G1D | **112.50 ms** | **114.93 ms** | 同轮 24 次；相对原 3F 均值下降 **10.7%** |

不同硬件、checkpoint、任务和缓存策略之间不能当作同轮消融相减。对原 3F 到第二轮 3F 的收益，仅有组合对照，不能给每项新增优化分摊未经测量的收益。

G1D 计时为 CUDA 同步的 `server.predict`，每次清空动作队列并生成全新 32 步动作；包括解码/预处理、VAE、条件准备、采样和动作后处理/CPU 返回。排除网络、payload deepcopy、reset、磁盘写入、模型加载、编译及预热。

输入为 6 个 held-out G1D 观测、3 个任务提示词、2 个位置，各使用 seeds 42/123，共 12 个 case。图像从已缩放的 384×320 验证集组合图重建，**不包含真实相机采集和全分辨率缩放负载**。测量中每 case 重复两次。运行时为 PyTorch 2.7.1+cu128 / CUDA 12.8；DiT、action model、VAE 在一张 RTX 4090，文本编码器放在另一张 GPU，不是两卡并行 DiT 加速。

另一次 WebSocket 测试为 12 次新动作段请求，平均 roundtrip 122.96 ms、P95 136.28 ms；该口径包含序列化/传输，不能与上述直接调用时间混用。

## 5. 正确性、执行证据与尚未证明的部分

G1D 的全部 12 个配对 case 中，原 3F 与第二轮 3F 的完整 `(32,80)` 归一化动作及 `(32,20)` 物理动作均逐位相同。计数器验证：

- 3 次完整 Transformer、7 次残差复用、10 次输出头求值；
- 1 次编译 prefill、3 次编译 joint 调用，无编译失败；
- time cache 每请求 13 hits、0 misses；轻量路径 7 次执行、0 fallback；
- 第二轮主 GPU 峰值 allocated 14.28 GiB。

这证明本轮执行了指定优化路径，且在这些样例上第二轮保持原 3F 输出；**不证明 3F 与完整 10F 等价，也不证明真机闭环成功率不下降**。G1D 正式 3000 步权重的离线验证结果也不能自动当作 3F 的质量验证。

没有通过修改权重、量化、降低输入分辨率或缩短动作段取得上述结果。手写 Triton RoPE / RMSNorm+RoPE 曾在单算子测试加速，但没有证明完整请求收益，因此 3F 配置保留 `rope_kernel: torch`；Q/K/V GEMM 融合及更激进 autotune 也未作为有效收益采用。

## 6. 配置与复现检查

已测配置名称：`deploy_realtimewam_3f_v2.yaml`。关键设置如下（仅为实现对接说明，不是当前 main 可独立运行的配置）：

```yaml
inference:
  denoise_steps: 10
  denoise_mode: sync
optimization:
  residual_cache:
    enabled: true
    refresh_steps: [0, 5, 9]
  observation_prefix_cache:
    enabled: true
    compile_enabled: true
    compile_mode: reduce-overhead
    verify_invariants: false
  fixed_schedule_time_cache:
    enabled: true
  rope_cache:
    enabled: true
  lightweight_residual_reuse:
    enabled: true
  dit_cache:
    enabled: false
  rope_kernel: torch
```

完整复现还需实现快照、完整配置、checkpoint、训练时 normalization、相机布局及部署动作契约。G1D checkpoint SHA256 为 `d40f61111f86910b7a98520edb3e396e522f33f83593e8baf34ccba94a94b2e1`。

迁移验证应先用完整 10F 建立参考，再测原 3F 的质量变化，再比较第二轮 3F；保存完整动作数组、每请求时间分布和缓存计数。每个正式计时请求强制新生成，避免把动作队列取值测成模型推理。缓存冷启动、首次编译及新形状重新编译单独报告。

随文提供 [G1D 已测摘要](inference-evidence/g1d-3f-summary.json) 和 [v3 已测摘要](inference-evidence/v3-summary.json)。它们由本地结果提取，去除了机器路径和服务地址；本次文档整理未重新运行 GPU 基准。
