# Action-only No-WM OOM 调试结论

日期：2026-09-25

## 结论

这次 `lambda_video=0.0`、batch size 已经降到 1 后仍然 OOM，主要不是 T5 text encoder，也不是单纯 batch size 太大，而是 action-only/no-WM 的训练面没有真正只训练 action 分支。

原始日志里模型参数统计显示：

```text
video_backbone : trainable=4999.8M
action_backbone: trainable=1021.0M
Architecture  : trainable=6021.2M
```

也就是说，即使 `lambda_video=0.0`，训练仍把约 5B 的 video backbone 参数放进 optimizer/ZeRO 管理范围。OOM 堆栈出现在 `torch._foreach_sqrt` / `foreach_tensor_sqrt_cuda`，对应 AdamW/DeepSpeed optimizer step 的中间张量和优化器状态峰值。`lambda_video=0` 只让最终 video loss 权重为 0，并不会自动冻结 video backbone，也不会阻止原始 `BaseWAMArchitecture.compute_loss()` 计算 video target 和 video MSE。

## 已做的本地修复

所有修改都限制在 `train-on-robotwin/` 下，没有改上游 `OpenWAM/`。

1. 新增 `train-on-robotwin/code/robotwin_oom_patches.py`
   - 在创建 optimizer 前支持 `training.freeze_video_backbone=true`，将 `architecture.video_backbone.requires_grad_(False)`，并从 optimizer 参数组中移除约 4,999,787,712 个 video backbone 可训练参数。
   - patch `OpenWAMTrainer.build_optimizer()`，默认使用 `torch.optim.AdamW(..., foreach=False)`，避免 foreach AdamW 在 GPU 上产生较大的临时峰值。
   - 当 `training.skip_zero_weight_video_loss=true` 且 `lambda_video=0` 时，让 video loss 直接返回 GPU scalar zero，跳过零权重 video MSE 的额外计算。

2. 更新 `train-on-robotwin/code/train_robotwin_split.py`
   - 在构造 `OpenWAMTrainer` 前安装上面的本地 patch。
   - DeepSpeed plugin 增加 `offload_param_device`、`zero3_init_flag` 配置透传。
   - 修正 `zero3_init_flag` 的字符串布尔解析，避免 `bool("false") == True`。

3. 更新 `train-on-robotwin/code/run_formal_action_only_8gpu_bs32.sh`
   - `batch_size=1`，`gradient_accumulation_steps=4`，保持 8 卡 effective global batch size = 32。
   - `training.zero_stage=2`。
   - `training.offload_optimizer_device=cpu`。
   - `+training.adamw_foreach=false`。
   - `+training.freeze_video_backbone=true`。
   - `+training.skip_zero_weight_video_loss=true`。
   - `model.architecture.attention_mask_mode=action_sees_video`，让 action 可以看 video token，但 video token 不反向依赖 action token。
   - 启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
   - 移除了脚本里的硬编码 W&B key，只在环境变量 `WANDB_API_KEY` 已存在时执行 `wandb login`。

4. 新增 `train-on-robotwin/code/run_smoke_action_only_small_model.sh`
   - 用更小的 `wan21_vace_1_3b` 路径做 action-only smoke 测试入口。
   - 默认同样使用 `batch_size=1`、`grad_accum=4`、ZeRO-2、CPU optimizer offload、禁用 AdamW foreach、冻结 video backbone。

## 验证结果

静态检查通过：

```bash
python -m py_compile train-on-robotwin/code/train_robotwin_split.py train-on-robotwin/code/robotwin_oom_patches.py
bash -n train-on-robotwin/code/run_formal_action_only_8gpu_bs32.sh train-on-robotwin/code/run_smoke_action_only_small_model.sh
```

5B 模型、8 卡、cached T5、未冻结 video backbone，但禁用 foreach/跳过 zero-weight video loss 的 60 step 验证可跑通：

```text
log: train-on-robotwin/code/audit/rank_logs/debug_patch60_20260925_160154/rank0.log
step 1 : allocated=12.78GiB reserved=23.77GiB max_allocated=14.87GiB
step 60: allocated=17.58GiB reserved=23.90GiB max_allocated=19.67GiB
steps/sec ~= 0.18
```

5B 模型、8 卡、cached T5、冻结 video backbone 的 60 step 验证可跑通，并明显更快：

```text
log: train-on-robotwin/code/audit/rank_logs/debug_freeze60_20260925_160756/rank0.log
frozen_params=4999787712
step 1 : allocated=12.66GiB reserved=15.38GiB max_allocated=14.58GiB
step 60: allocated=16.02GiB reserved=18.38GiB max_allocated=17.94GiB
steps/sec ~= 0.72
```

5B 模型、8 卡、cached T5、冻结 video backbone 的 200 step 验证可跑通，8 个 rank 都到达 step 200，无 CUDA OOM / traceback：

```text
log: train-on-robotwin/code/audit/rank_logs/debug_freeze200_20260925_161102/rank0.log
frozen_params=4999787712
AdamW foreach=False
step 1  done: allocated=12.66GiB reserved=15.38GiB max_allocated=14.58GiB
step 50 done: allocated=15.45GiB reserved=17.47GiB max_allocated=17.37GiB
step 100 done: allocated=18.29GiB reserved=20.38GiB max_allocated=20.21GiB
step 150 done: allocated=21.14GiB reserved=23.35GiB max_allocated=23.06GiB
step 200 done: allocated=23.99GiB reserved=26.36GiB max_allocated=25.91GiB
steps/sec ~= 0.55
train/loss_video = 0
```

## 仍需注意

200 step 验证虽然通过，但 `allocated` 仍随 step 从约 12.66GiB 增到约 23.99GiB。当前没有看到 Python 层保存 loss graph 的明显代码路径，比较像动态 shape / CUDA allocator / DeepSpeed optimizer 工作区逐步触发更大缓存；但 50k 长跑前仍建议保留：

```bash
OPENWAM_STEP_DIAG=1
OPENWAM_STEP_DIAG_EVERY=10
```

如果长跑中显存继续线性上涨到危险区，下一步应继续做更强的 action-only 路径：让 `lambda_video=0` 时不返回 video prediction，或者在 architecture forward 内给 video backbone 加 `torch.no_grad()`/显式 detach video tokens。但这会改变 action 分支接收 video token 的梯度路径，需要单独核对模型设计。

## 推荐正式运行

```bash
cd /home/chw/code/packages/OpenWAM
bash train-on-robotwin/code/run_formal_action_only_8gpu_bs32.sh
```

如需指定 W&B key，在运行前放到环境变量里：

```bash
export WANDB_API_KEY=...
```
