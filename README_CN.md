# LumbarSR 挑战赛

**腰椎CT超分辨率结构重建挑战赛**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 概述

LumbarSR是一个医学影像挑战赛，专注于腰椎CT图像的超分辨率重建。目标是从临床CT扫描重建高分辨率光子计数CT（PCCT）图像。

## 任务描述

给定使用**软组织重建核**采集的两种不同视野（FOV）的临床CT图像：
- **小视野** (面内分辨率195um)
- **大视野** (面内分辨率586um)

重建对应的高分辨率Micro-PCCT图像（105um各向同性分辨率）。

## 数据集

### 下载

**[Google Drive 链接]** (即将发布)

### 数据划分

| 划分 | 样本数 | 临床CT | Micro-PCCT |
|------|--------|--------|------------|
| 训练集 | 30 | 配对 | 配对 |
| 验证集第一阶段 | 5 | 配对 | 配对 |
| 验证集第二阶段 | 5 | 提供 | 隐藏 |

### 输入数据（临床CT）

每个样本提供4个软组织核序列：

| 序列 | 面内分辨率 | 层厚 | 视野 |
|------|-----------|------|------|
| 195X_195Y_500Z_S | 195um | 500um | 小 |
| 195X_195Y_1000Z_S | 195um | 1000um | 小 |
| 586X_586Y_500Z_S | 586um | 500um | 大 |
| 586X_586Y_1000Z_S | 586um | 1000um | 大 |

### 金标准（Micro-PCCT）

- 分辨率：105um各向同性
- 格式：NIfTI (.nii.gz)
- 数据类型：int16（Hounsfield单位）

## 评估指标

- **PSNR**（峰值信噪比）
- **SSIM**（结构相似性指数）
- **MAE**（平均绝对误差）
- **NCC**（归一化互相关）

指标在骨窗下计算（WC=400, WW=1800）。

## 提交

详情即将公布。

## 时间线

- **第一阶段**：使用配对数据进行训练和验证
- **第二阶段**：在隐藏测试集上进行最终评估

## 引用

如果您使用此数据集，请引用：

```bibtex
@misc{lumbarsr2026,
  title={LumbarSR: Lumbar Vertebral Super-Resolution Structure Reconstruction Challenge},
  year={2026},
  url={https://github.com/frankzhangrp/LumbarSR-Challenge}
}
```

## 联系方式

- 邮箱：zhangrp@sjtu.edu.cn
- GitHub：[@frankzhangrp](https://github.com/frankzhangrp)

## 许可证

本数据集采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证发布。
