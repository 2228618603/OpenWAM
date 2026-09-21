# RoboTwin Hard8 评测全流程与数据说明

本文只解释这次已经跑完的 Hard8 评测，不讨论重新训练。

## 1. 评测全流程：按数据流看

这次评测的数据流是：

```text
OpenWAM checkpoint
  -> 启动 OpenWAM policy server
  -> RoboTwin 仿真环境启动一个任务
  -> 仿真器渲染当前相机图像和机器人状态
  -> RoboTwin client 把观测发给 OpenWAM server
  -> OpenWAM server 返回动作
  -> RoboTwin 仿真器执行动作
  -> 判断这一条 episode 成功或失败
  -> 写入 episode-level JSONL
  -> 汇总 success@k / success rate
```

更通俗地说：

```text
模型看仿真画面 -> 模型给动作 -> 仿真环境执行动作 -> 记录成功/失败
```

这次评测没有把 RoboTwin 训练集喂给模型，也没有重新训练模型。

## 2. 本次实际用到的东西

本次用到的是三类东西：

### 2.1 OpenWAM policy checkpoint

路径：

```text
/mnt/data/chw/model/openwam_ckpt/openwam_study/video_backbone/
```

包含 5 个已经 fine-tuned 好的 RoboTwin checkpoint：

```text
wan21_vace_1_3b
cosmos25
cosmos3
wan22_ti2v_5b
wan21_i2v_14b
```

这些是“已经训练好的模型”，不是 RoboTwin 数据集。

### 2.2 RoboTwin 仿真环境代码和任务资产

本次实际跑的是兼容副本：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat_copy
```

它负责：

```text
创建任务场景
加载物体/机器人
运行 SAPIEN/MPLib/cuRobo 等仿真和规划逻辑
执行动作
判断成功失败
```

这部分是“仿真环境”，不是训练数据集。

### 2.3 评测结果

最终结果在：

```text
/home/chw/code/packages/OpenWAM/chw-code/robotwin_legacy_compat/result/hard8_stage8_20260917_072352
```

核心文件：

```text
episodes/*.jsonl
summaries/per_task_passk.csv
summaries/backbone_average_passk.csv
notes/stage9-hard8-analysis.md
```

这些是评测输出，不是训练数据集。

## 3. 是否下载了 RoboTwin 数据集？

没有。

这次 Hard8 评测没有下载 RoboTwin2.0 训练数据集，也没有复制训练集到：

```text
/mnt/data/chw/fastwam/data/robotwin2.0/robotwin2.0/
```

我查过本机常见位置，没有发现这 8 个任务的 `episode*.hdf5` 训练数据，也没有发现对应的 `aloha-agilex*.zip` 数据包。

## 4. 如果下载 RoboTwin2.0 数据集，是什么格式？

OpenWAM 的 RoboTwin dataloader 期望的是 RoboTwin 2.0 的 HDF5 数据格式，目录大致是：

```text
robotwin2.0/
  dataset/
    <task_name>/
      aloha-agilex_clean_50/
        data/
          episode0.hdf5
          episode1.hdf5
          ...
        scene_info.json
        instructions/
      aloha-agilex_randomized_500/
        data/
          episode0.hdf5
          ...
```

也就是说，如果以后下载训练数据，主要会是：

```text
episode*.hdf5
```

不是 LeRobot 3.0，也不是 LeRobot 2.1。

本次评测没有用 LeRobot 数据格式。

## 5. 仿真环境和数据集是不是一回事？

不是。

可以这样区分：

```text
仿真环境 = 现场搭一个虚拟世界，让模型实时操作
训练数据集 = 过去录好的 demonstrations / episodes
```

本次评测用的是：

```text
RoboTwin 仿真环境 + 已训练好的 OpenWAM checkpoint
```

本次评测没有用：

```text
RoboTwin2.0 HDF5 训练数据集
LeRobot 数据集
重新训练流程
```

## 6. A 卡和 H 卡兼容性问题怎么理解？

别人说的“A 卡和 H 卡上 RoboTwin 数据集/仿真环境不兼容”，通常要分开看：

### 6.1 数据集兼容性

如果只是读取 `episode*.hdf5` 训练数据，通常和 GPU 型号关系不大。

HDF5 是文件格式，主要依赖 CPU、磁盘和 Python/HDF5 库。

### 6.2 仿真环境兼容性

真正容易受 GPU/驱动影响的是仿真环境：

```text
SAPIEN
Vulkan / EGL
cuRobo
CUDA
显卡驱动
offscreen rendering
```

本次评测是实时跑 RoboTwin 仿真，所以主要风险在这一层。

### 6.3 本次实际情况

这次已经在当前机器上完整跑完：

```text
40 / 40 jobs
1280 / 1280 episodes
```

说明当前机器上的仿真链路至少对这次 Hard8 `demo_clean` 评测是可用的。

## 7. 一句话总结

这次做的是：

```text
用已经训练好的 OpenWAM 模型，在 RoboTwin 仿真环境里实时跑 8 个任务评测。
```

这次没有做的是：

```text
下载 RoboTwin2.0 训练数据集、读取 LeRobot 数据、或者重新训练模型。
```

