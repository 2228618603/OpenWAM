# 联合训练显存随 step 增长调试记录

时间：2026-09-26

## 结论

这次问题不是“video backbone 不该训练”。联合训练里 video backbone 和 action DiT 都应该更新；本次配置下 T5 不加载，文本来自 cached text embeddings，所以显存比加载 T5 时低很多是正常的。

真正导致“每个 step 后显存 baseline 越来越高”的主因，是 `DualSystemMoTDriver` 里整层 MoT checkpoint：

```text
pre_attn_at_layer -> mixed_attention -> post_attn_at_layer
```

这段旧实现用 `torch.utils.checkpoint(..., use_reentrant=False)` 包住一个会浅拷贝并改写 `vstate` / `astate` 的闭包。在 DeepSpeed/ZeRO 训练下，这条 non-reentrant checkpoint 路径会在 C++ autograd/checkpoint 侧逐步保留状态，表现为 Python 侧看不到明显 tensor/list 泄漏，但 `torch.cuda.memory_allocated()` 每隔一段 step 就继续抬高，最终 OOM。

已经修复：

- 默认禁用 MoT 的整层外层 checkpoint。
- 仍保留 `mot_checkpoint_mixed_attn=true` 的小范围 mixed attention checkpoint。
- 保留训练循环末尾的显式 `del metrics, losses, loss, grad_norm, batch`，避免 Python 局部变量跨 step 多活一轮。
- AdamW 设置 `foreach=False`，避免 `_foreach_sqrt` 在 optimizer step 里额外申请大块临时显存。

代码位置：

```text
OpenWAM/openwam/model/architectures/dual_system/mot_driver.py
OpenWAM/openwam/train/openwam_trainer.py
```

## 不是哪类问题

这次不是“每个样本算完 loss 后没有把 batch 完整释放”这么简单。

我做了几个排查：

- `optimizer.zero_grad(set_to_none=True)` 后，`grad_tensors=0`，说明不是 `.grad` 没清。
- DeepSpeed/Adam optimizer state 在 step 2 后固定约 16.82 GiB，后面不继续增长，说明不是优化器状态逐步膨胀。
- `gc.get_objects()` 里 Python 可见 CUDA tensor 的大 shape/top list 基本不随 step 变化，说明不是 Python list/dict 持有大量历史 tensor。
- 强制 `gc.collect()` 和 `torch.cuda.empty_cache()` 只能降 `reserved`，不能阻止 `allocated` 增长，说明不是 allocator cache 的假增长。

所以更准确的理解是：上一轮普通 Python batch/loss 不是主因；主因在 autograd/checkpoint 内部保存的重算上下文没有按 step 稳定回落。

## 神经网络计算过程里发生了什么

正常一次训练 step 是：

1. forward：video/action 输入经过 DiT，每层产生激活，用来给 backward 算梯度。
2. loss：得到 video loss 和 action loss；action-only 时 `lambda_video=0`，但 video backbone 仍参与 action loss 的前向和梯度链路。
3. backward：根据 loss 反向传播，释放大部分中间激活。
4. optimizer step：用梯度更新 video/action 参数，同时维护 Adam 的一阶/二阶动量。
5. zero_grad：清掉梯度，进入下一 step。

activation checkpoint 的作用是：forward 时不保存某些中间激活，backward 时重新跑一遍局部 forward 来省显存。它不是“免费释放”，而是把“存激活”换成“反向时重算”。

这次出问题的是 MoT 外层 checkpoint 包住的不是一个纯函数，而是一个会围绕 `vstate.hidden_states` 和 `astate.payload.x_action` 做状态搬运的闭包。non-reentrant checkpoint 会在 autograd graph 里管理更多保存/重算信息；和 DeepSpeed/ZeRO 的参数分片、梯度同步、bf16/fp32 optimizer state 放在一起后，显存没有在每 step 后回到稳定 baseline。

## 对照实验

有问题的旧路径：bs=2、grad_accum=2、action-only、8 卡，`training.use_gradient_checkpointing=true`，会持续增长：

| step | after_step_cleanup allocated |
| --- | --- |
| 25 | 26.71 GiB |
| 50 | 29.11 GiB |
| 75 | 34.82 GiB |
| 100 | 37.23 GiB |
| 150 | 45.34 GiB |
| 200 | 53.45 GiB |
| 250 | 61.57 GiB |

关闭 MoT 外层 checkpoint 后，同配置 120 step 稳定：

| step | after_step_cleanup allocated |
| --- | --- |
| 1 | 17.05 GiB |
| 2 | 21.00 GiB |
| 3 | 22.66 GiB |
| 25 | 22.66 GiB |
| 50 | 21.00 GiB |
| 75 | 22.66 GiB |
| 100 | 21.00 GiB |

修复后保持正式配置 `training.use_gradient_checkpointing=true` 再跑 120 step，也稳定：

| step | after_step_cleanup allocated |
| --- | --- |
| 1 | 17.05 GiB |
| 2 | 21.00 GiB |
| 3 | 22.66 GiB |
| 25 | 22.66 GiB |
| 50 | 21.00 GiB |
| 75 | 22.66 GiB |
| 100 | 21.00 GiB |

日志：

```text
train-on-robotwin/code/audit/launch_action_only_bs2_acc2_no_outer_ckpt_tmux.log
train-on-robotwin/code/audit/launch_action_only_bs2_acc2_fixed_outer_ckpt_tmux.log
```

## 当前处理建议

1. 联合训练继续保留 video backbone 可训练。
2. 正式训练仍可传 `training.use_gradient_checkpointing=true`，但 dual-system MoT 不再默认使用整层外层 checkpoint。
3. 不建议重新打开 `OPENWAM_MOT_OUTER_CHECKPOINT=1`，除非只是为了复现实验；这会回到有显存增长风险的旧路径。
4. 如果后续完整 WM 加 `lambda_video=1` 后显存仍高，那属于模型/bs 本身的容量问题，应优先用 bs=2、梯度累积=2，或进一步做 ZeRO/offload/更小模型；不要再用这个外层 MoT checkpoint 当省显存方案。

一句话版：video/action 训练方向没错，T5 不加载也没问题；OOM 的关键 bug 是 MoT 整层 checkpoint 在 DeepSpeed/ZeRO 下造成跨 step 显存 baseline 增长。
