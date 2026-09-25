# OpenWAM RoboTwin Action-Only OOM 与 T5 离线预编码实施说明

记录时间：2026-09-25

本文档用于在清空上下文、换终端后继续排障和实现。当前目标是：在 RoboTwin no-wm/action-only 训练中，先离线预编码 prompt 的 T5 text embedding，再让训练阶段直接读取缓存，从而避免训练进程加载 T5，降低显存占用。

约定缓存落盘位置：

- context embedding cache 放到 `/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50/`
- 不放到 OpenWAM 代码仓库下，避免污染 git、拖慢代码同步，也避免占用代码盘空间。
- 代码仓库中只保留预编码脚本、读取逻辑和必要的说明文档。

## 1. 当前训练场景

当前训练是 OpenWAM + RoboTwin，使用 Wan2.2-TI2V-5B video backbone 和 dual_system joint_self_attn 架构，只训练 action loss：

- `model=dual_system`
- `model/video_backbone=wan22_ti2v_5b`
- `model.architecture.variant=joint_self_attn`
- `model.architecture.attention_mask_mode=mutual`
- `training.lambda_video=0.0`
- `training.lambda_action=1.0`
- 8 卡训练
- 单卡 `training.batch_size=2`
- `training.gradient_accumulation_steps=1`
- 全局 batch size 为 16
- DeepSpeed ZeRO stage 2
- seed 固定为 `project.seed=42`
- `dataset_num_workers=0`

启动脚本位置：

- `train-on-robotwin/code/run_formal_action_only_8gpu_bs32.sh`

注意：脚本名里仍有 `bs32`，但当前实际配置已经是单卡 batch size 2，总 batch size 16。

## 2. 已遇到的问题与结论

### 2.1 OOM 不是通信超时本身导致

最新诊断训练在 step 171 附近失败。最先失败的是 rank 1 和 rank 7，rank 0 的 Gloo/NCCL barrier timeout 是后续连锁反应。

关键错误位置：

- `accelerator.backward(loss)`
- DeepSpeed ZeRO-2 optimizer step
- `deepspeed/runtime/zero/stage_1_and_2.py`
- OOM 行为是 ZeRO-2 在 flatten/partition averaged gradients 时尝试额外分配约 2.80 GiB

日志显示：

- GPU 1 当时剩余约 2.25 GiB
- PyTorch allocated 约 68 GiB
- PyTorch reserved 约 74.5 GiB
- 因额外 2.80 GiB 分配失败导致 OOM

结论：真正根因是 rank 1 / rank 7 的 CUDA OOM；rank 0 的分布式超时只是其他 rank OOM 后的副作用。

### 2.2 显存是单次训练内部逐 step 增长，不是退出后没释放

训练挂掉后检查 `nvidia-smi`，8 张卡都只有约 537-550 MiB 占用，没有残留 `torchrun` / `deepspeed` / `train_robotwin` 训练进程。

所以不是“上一次训练进程退出后显存没释放”。

但在同一次训练 run 内部，显存确实持续上涨。rank 1 诊断日志示例：

- step 1 done：allocated 31.80 GiB, reserved 51.80 GiB
- step 20 done：allocated 35.56 GiB, reserved 51.83 GiB
- step 80 done：allocated 47.41 GiB, reserved 57.54 GiB
- step 120 done：allocated 55.29 GiB, reserved 66.01 GiB
- step 160 done：allocated 63.17 GiB, reserved 74.49 GiB
- step 170 done：allocated 65.14 GiB, reserved 74.50 GiB
- step 171 exception：allocated 66.75 GiB, reserved 74.50 GiB

每个 batch 的样本形状稳定：

- 单卡 batch size 2
- action shape 为 `(32, 80)`
- action_mask shape 为 `(32, 80)`
- video_mask shape 为 `(9,)`
- proprio shape 为 `(1, 80)`
- prompt 是字符串

结论：目前更像训练图、DeepSpeed ZeRO-2 梯度/optimizer buffer、视频 backbone/action joint forward 侧的显存压力或保留，而不是某个异常超大 batch 或 DataLoader worker 堆积。

### 2.3 T5 确实加载了，而且当前 action-only 训练仍会使用它

日志明确出现：

- `Loading models from: "/mnt/data/chw/model/openwam/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"`
- `model_name: "wan_video_text_encoder"`
- `model_class: "openwam.model.video_backbone.wan.models.text_encoder.WanTextEncoder"`
- `Using wan_video_text_encoder from "...models_t5_umt5-xxl-enc-bf16.pth"`

配置里有：

- `freeze: video_backbone.text_encoder`

但 freeze 只表示不训练参数，不表示不加载、不占显存。当前代码路径里，Wan backbone 会把 `text_encoder` 注册为子模块，随后设备迁移会把它放到 GPU。

关键代码位置：

- `OpenWAM/openwam/model/video_backbone/wan_backbone.py`
  - `__init__` 中会注册 `text_encoder`
  - `preprocess_input_for_train()` 中会调用 `wan_encode.encode_text(... text_encoder=self.text_encoder ...)`
- `OpenWAM/openwam/model/video_backbone/base.py`
  - `set_dtype_device()` 默认 `self.to(dtype=dtype, device=device)`，会移动注册子模块
- `OpenWAM/openwam/model/architectures/base.py`
  - `prepare_inputs()` 被 `@torch.no_grad()` 装饰，但它会无条件调用 `self.preprocess(...)`
- `OpenWAM/openwam/train/openwam_trainer.py`
  - `compute_loss()` 先调用 `architecture.prepare_inputs(batch)`，再把 `lambda_video/lambda_action` 传入 `compute_loss`

因此，即使 `lambda_video=0.0`，当前训练仍然会在每个 step 预处理视频与文本，并调用 T5 得到 `context` / `seq_lens`。这部分在 `torch.no_grad()` 下执行，不构建梯度图，但会占用 T5 模型常驻显存和文本编码临时显存。

## 3. 现在准备怎么解决

核心思路：把 T5 的输出提前离线算好，训练时直接读取缓存的文本 embedding。

当前 action-only 训练不是完全不需要文本。action backbone 在 joint self-attn 架构下仍然会用 `context` / `seq_lens` 作为文本条件。所以不能直接删除 T5 输出，否则要么前向缺字段，要么训练语义改变。

正确方向是：

1. 离线遍历训练集 prompt。
2. 用当前完全相同的 tokenizer、T5 权重、dtype、prompt 文本模板、截断/padding 逻辑，生成 `context` 和 `seq_lens`。
3. 将结果保存为可索引缓存。
4. 训练时不加载 T5，而是从缓存读取 `context` / `seq_lens` 并填入 `prepare_inputs()` 的结果。
5. 先做小步验证，确保缓存输出和在线 T5 输出数值一致，再跑短训练，最后再跑正式训练。

这样理论上可以省掉训练进程中的 T5 常驻显存，也能去掉每个 step 的 T5 编码临时显存。

缓存目录建议结构：

```text
/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50/
├── manifest.json
├── index.jsonl
└── embeddings/
    ├── <prompt_hash>.pt
    ├── <prompt_hash>.pt
    └── ...
```

每个 `.pt` 文件建议至少包含：

- `context`: CPU tensor，建议 bf16，形状与在线 T5 输出一致。
- `seq_len` 或 `seq_lens`: 当前 prompt 的有效 token 长度。
- `prompt`: 原始 prompt 文本，用于人工核查和 hash 冲突防御。
- `meta`: T5 checkpoint、tokenizer、dtype、hash 规则、缓存版本等信息。

这个缓存会占磁盘。粗略估算：如果 `context` 是 `[seq_len, 4096]` bf16，每个 token 约 8 KB；一个 128-token prompt 约 1 MB。若有 1 万个 unique prompt，约 10 GB；5 万个 unique prompt，约 50 GB。实际大小取决于 prompt 去重率和序列长度，所以应该放到空间更宽裕的 `/mnt/data` 盘。

## 4. 分 Stage 实施计划与检查点

### Stage 0：冻结现状和基线

目的：保证后续改动前后有可比较基线。

要做的事：

- 保存当前训练配置、启动脚本和最新 OOM 日志路径。
- 确认当前训练仍然会加载 T5。
- 记录当前 OOM step 和显存曲线。

重点日志：

- `train-on-robotwin/code/audit/tmux_logs/openwam_diag_bs16_rerun.log`
- `train-on-robotwin/code/audit/rank_logs/formal_action_only_no_wm_50k_8gpu_20260925_025956/`

检查点：

- `rg "models_t5|wan_video_text_encoder|lambda_video|lambda_action|zero_stage|batch_size" <log>` 能看到 T5 加载和 action-only 配置。
- `nvidia-smi` 在训练退出后无残留大显存占用。
- 明确记录 OOM 发生在 step 171 附近、DeepSpeed ZeRO-2 backward/step 阶段。

### Stage 1：弄清 T5 输出的数据结构

目的：确认在线 T5 输出究竟有哪些字段、shape、dtype、device，缓存必须完全兼容这些字段。

要做的事：

- 在 `WanVideoBackbone.preprocess_input_for_train()` 或 `BaseWAMArchitecture.prepare_inputs()` 增加临时调试打印，只跑 1-2 step。
- 打印：
  - `context.shape`
  - `context.dtype`
  - `context.device`
  - `seq_lens.shape`
  - `seq_lens.dtype`
  - `seq_lens`
  - 是否存在 `context_mask`
- 确认 batch 内不同 prompt 长度时 `context` 是否 padding 到同一长度。

检查点：

- 能明确得到 `context` 的 shape，例如 `(B, S, D)`。
- 能明确 `D` 是否为 4096。
- 能明确 `seq_lens` 是否是每个样本一个长度。
- 清楚缓存需要保存 `context` 和 `seq_lens`，以及可能需要 `context_mask`。

成功标准：

- 用一个 batch 的在线 T5 输出形成样例，后续缓存读取能复现这个结构。

### Stage 2：设计缓存 key 和缓存格式

目的：确保每个训练样本能稳定读到正确 text embedding，且多次训练可复现。

推荐 key：

- 优先使用 prompt 文本的 hash 作为 key。
- 如果同一个 prompt 会反复出现，hash prompt 能最大化复用。
- 建议 hash 输入包含：
  - prompt 字符串
  - tokenizer 路径或版本标识
  - T5 checkpoint 路径或文件名
  - 最大长度/截断策略

推荐缓存格式：

- 简单稳妥版：每个 prompt 一个 `.pt` 文件，内容为：
  - `context`: CPU tensor, dtype 建议 `torch.bfloat16`
  - `seq_len`: CPU tensor 或 int
  - `prompt`: 原始字符串
  - `meta`: tokenizer/model/checkpoint/hash 信息
- 可扩展版：使用 shard，例如每 N 个 prompt 存一个 `.pt`，再维护 index json/jsonl。

建议先做简单稳妥版，因为当前目标是先跑通。

缓存根目录固定使用：

- `/mnt/data/chw/model/wm-function/cache/robotwin_text_embeddings/wan22_ti2v_5b_umt5_xxl_bf16_clean50/`

不要把 embedding cache 放在：

- `train-on-robotwin/code/`
- `train-on-robotwin/note/`
- `OpenWAM/`
- 任何会被 git 跟踪或容易被 rsync/scp 代码时误带走的位置

检查点：

- 同一个 prompt 多次计算得到同一个 key。
- 缓存目录里能按 key 找到对应 `.pt`。
- 读回缓存后，`context.dtype`、`context.shape` 和在线 T5 输出一致。

成功标准：

- 随机抽 10 个 prompt，key 稳定，缓存可读，字段完整。

### Stage 3：写离线预编码脚本

目的：用现有 Wan tokenizer/T5 权重，对训练集所有 prompt 预编码。

建议脚本位置：

- `train-on-robotwin/code/preencode_robotwin_text_embeddings.py`

建议脚本功能：

- 复用当前 dataloader / split / dataset 逻辑，遍历训练样本。
- 收集 prompt，去重。
- 加载 Wan text encoder 和 tokenizer。
- 调用当前同一路径的 `wan_encode.encode_text(...)`。
- 将输出转到 CPU 后保存。
- 写入 manifest，例如：
  - prompt 数量
  - unique prompt 数量
  - cache 版本
  - model path
  - tokenizer path
  - dtype
  - max sequence length

注意事项：

- 离线预编码可以用单卡或 CPU/GPU 混合，不需要 8 卡。
- 保存时必须 detach + cpu，避免把计算图或 GPU tensor 状态带入缓存。
- 如果 T5 输出为 bf16，优先按 bf16 存，节省磁盘和加载内存。

检查点：

- 脚本能对小 subset 例如 20 个 prompt 完成缓存。
- manifest 里 unique prompt 数量合理。
- 缓存文件数或 shard 内条目数与 unique prompt 数一致。
- 预编码期间显存会加载 T5，这是预期；训练阶段才不加载 T5。

成功标准：

- 对完整训练集跑完后，所有训练样本 prompt 都能命中缓存。

### Stage 4：给 dataset 或 prepare_inputs 接入缓存

目的：训练时从缓存读取 text embedding，跳过 T5 encode。

推荐改法 A：在 dataset 层返回缓存字段。

- dataset sample 里新增：
  - `text_context`
  - `text_seq_len`
- `BaseWAMArchitecture.prepare_inputs()` 检测 batch 是否已经有 `text_context`。
- 如果有，就跳过 `self.preprocess(... text=all_prompts ...)` 中的 T5 编码逻辑，或者走一个新的 no-text-encoder preprocess 分支。

推荐改法 B：在 Wan backbone 的 `preprocess_input_for_train()` 增加可选参数。

- 接收 `text_context` / `text_seq_lens`。
- 如果提供，则直接使用它们，不调用 `wan_encode.encode_text()`。
- 仍然执行视频/VAE相关预处理。

短期更稳妥的是改法 B，因为最接近当前 `context/seq_lens` 的生成位置。但如果目标是后续连视频/VAE预处理也跳过，dataset/architecture 层会更方便。

检查点：

- 加一条日志：使用缓存时打印一次 `Using cached text embeddings; skipping Wan T5 encode`。
- 运行 1 step 时日志里不应出现 T5 每步 encode 的调试打印。
- 如果仍加载 T5，此阶段也可以先允许加载但不调用，用来验证数值；真正不加载放到 Stage 6。

成功标准：

- 在线 T5 输出与缓存输出在同一 batch 上：
  - shape 完全一致
  - seq_lens 完全一致
  - bf16 下误差为 0 或非常小
- 训练 1-5 step loss 能正常产生。

### Stage 5：数值一致性对齐测试

目的：确认缓存没有改变训练输入语义。

要做的事：

- 固定 seed。
- 选一个固定 batch。
- 跑在线 T5 路径，保存 `context_online` / `seq_lens_online`。
- 跑缓存路径，保存 `context_cached` / `seq_lens_cached`。
- 比较：
  - shape
  - dtype
  - seq_lens
  - max abs diff
  - mean abs diff

检查点：

- `seq_lens_online == seq_lens_cached`
- `context_online.shape == context_cached.shape`
- 如果缓存存 bf16，理论上读回应该完全一致；如果中间有 dtype 转换，误差也应非常小。

成功标准：

- 对固定 batch，缓存路径能复现在线路径的文本条件。
- 开启缓存前后，前几个 step loss 数值变化只来自随机采样/训练状态，而不是明显异常。

### Stage 6：训练阶段真正不加载 T5

目的：减少训练进程常驻显存。

要做的事：

- 在配置里增加一个开关，例如：
  - `training.use_cached_text_embeddings=true`
  - `training.text_embedding_cache_dir=...`
  - `model.video_backbone.load_text_encoder=false`
- 修改 Wan backbone 构建逻辑：
  - 如果使用缓存，并且训练阶段不需要在线文本编码，则不加载 `text_encoder` 权重。
  - `self.text_encoder` 可以为 `None`，但必须保证使用缓存路径时不会调用 `wan_encode.encode_text()`。
- 如果没有缓存命中，应该直接报错，不要悄悄 fallback 到在线 T5，否则会重新加载或调用 T5，显存优化失效。

检查点：

- 启动训练日志不再出现：
  - `Loading models from: "...models_t5_umt5-xxl-enc-bf16.pth"`
  - `Using wan_video_text_encoder`
- `nvidia-smi` 观察训练刚启动后的显存比原先低。
- 单步 `prepare_inputs()` 日志确认走 cached text path。

成功标准：

- 训练阶段完全不加载 T5。
- 1-10 step 能正常训练。
- 显存基线明显下降。

### Stage 7：短跑 OOM 回归测试

目的：确认去掉 T5 后是否缓解当前 OOM。

要做的事：

- 仍使用当前训练配置：
  - 8 卡
  - 单卡 bs=2
  - grad accum=1
  - no-wm/action-only
  - ZeRO stage 2
  - seed=42
- 启动短跑，例如 300 step。
- 保留现有 step 显存诊断。

检查点：

- step 1 / 20 / 80 / 120 / 160 / 170 的 allocated/reserved 与旧日志对比。
- 如果 171 后继续运行，说明至少缓解了触发点。
- 如果仍在接近 step 171 OOM，但初始显存下降，说明 T5 不是唯一问题。

成功标准：

- 至少跑过原失败点 step 171。
- 最好跑到 300 step 无 OOM。
- 如果仍 OOM，保留新日志继续判断是否需要 ZeRO-3、activation checkpoint/offload、跳过 video state、或改变 optimizer/参数冻结策略。

### Stage 8：进一步省显存：action-only 下跳过视频预处理和视频 state

目的：如果只去掉 T5 仍不够，进一步减少 video/VAE/Wan DiT 侧显存。

当前即使 `lambda_video=0.0`，joint_self_attn 仍会构造 video branch 并和 action branch 做 joint attention。所以只做 T5 缓存不一定足够。

后续可选方向：

- 对 action-only 训练增加专门路径：
  - 不编码视频 latents
  - 不跑 VAE
  - 不构造 video noise/video target
  - action branch 只使用 cached text context 和 proprio/action
- 或将 video branch 全冻结且尽可能 offload / checkpoint。
- 或换成 action-only architecture，避免完整 Wan video branch 参与 forward。

检查点：

- 关闭 video branch 后，`loss_action` 仍能计算。
- 输出动作预测 shape 正确。
- 显存显著下降，不再接近 70 GiB。

成功标准：

- action-only 训练能稳定跑通更长 step。
- loss 正常下降或至少无 NaN。
- 与原 joint_self_attn 语义差异已被明确接受。

## 5. 推荐最小实现顺序

建议按以下顺序推进，不要一口气大改：

1. 先做 Stage 1，打印在线 T5 输出结构。
2. 做 Stage 2 和 Stage 3，只对小 subset 预编码。
3. 做 Stage 4，让训练能读取缓存，但暂时仍允许 T5 加载，用于 A/B 对齐。
4. 做 Stage 5，证明缓存输出和在线输出一致。
5. 做 Stage 6，才关闭训练阶段 T5 加载。
6. 做 Stage 7，跑 300 step 对比显存曲线。
7. 如果仍 OOM，再进入 Stage 8。

## 6. 判断改动成功的总清单

最终实现应满足：

- 预编码脚本能完整生成 text embedding cache。
- cache manifest 记录模型、tokenizer、dtype、hash 规则和样本数量。
- 训练时所有 prompt 都能命中 cache。
- 缓存路径与在线 T5 路径的 `context` / `seq_lens` 一致。
- 训练日志不再加载 `models_t5_umt5-xxl-enc-bf16.pth`。
- 训练启动后初始显存低于原先。
- 训练能跑过旧失败点 step 171。
- 若仍 OOM，日志能说明瓶颈已不再是 T5，而是 video branch / DeepSpeed ZeRO-2 / optimizer buffer / activation retention。

## 7. 重要提醒

- 不要只把 `text_encoder=None`，但保留原本 `wan_encode.encode_text()` 调用；这会直接报错。
- 不要让 cache miss 时静默 fallback 到在线 T5；否则训练进程可能又加载 T5，显存优化失效。
- 不要改变 prompt 模板、tokenizer、T5 checkpoint、截断长度，否则训练输入语义会变。
- 固定 seed 可以保证初始化和随机采样更可复现，但离线缓存一致性主要依赖同一 tokenizer/T5/文本处理逻辑。
- 当前 `freeze` 策略不要顺手改。用户明确要求不要随意改变训练模型哪部分 freeze。
- wandb key 不要写进文档或日志。
