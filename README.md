# SPC Edge Detection
## Simplified Phase Congruency Edge Detection for Resource-Constrained Embedded Platforms

<p align="center">
  <img src="https://img.shields.io/badge/Platform-CYT4BB7-blue?style=flat-square&logo=arm" alt="Platform"/>
  <img src="https://img.shields.io/badge/Language-Python%203.8-yellow?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/C-IAR%20EW%20ARM%209.40-lightgrey?style=flat-square&logo=c" alt="C"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status"/>
</p>

<p align="center">
  <b>F1 = 0.958 (noise-free) ｜ F1 = 0.799 (σ=10) ｜ +139% over Canny under noise</b>
</p>

---

## 📖 简介 / Introduction

本项目为本科毕业论文《面向资源受限嵌入式平台的简化相位一致性边缘检测算法研究》的配套代码仓库。

This repository contains the source code for the paper:
> **"A Simplified Phase Congruency Edge Detection Algorithm for Resource-Constrained Embedded Platforms"**
> Liu Junjie, School of Electrical and Electronic Engineering, Anhui University of Information Engineering, 2026.

**核心思路 / Core Idea：**

原版相位一致性（Phase Congruency, PC）算法对光照变化具有理论最优的鲁棒性，但其多尺度多方向Log-Gabor滤波器组计算量过大，无法在英飞凌CYT4BB7（ARM Cortex-M7, 250 MHz, 768 KB SRAM）等无操作系统MCU平台上实时运行。

本文提出SPC（Simplified Phase Congruency），在保留PC核心鲁棒优势的前提下，将计算量降低至原版的约**10%**。

The original Phase Congruency algorithm provides theoretically optimal illumination robustness but is computationally prohibitive for MCU platforms without OS. SPC reduces computation to ~10% of the original while preserving the core robustness advantage.

---

## ✨ 主要创新点 / Key Contributions

| 创新点 | 描述 | 消融实验F1提升（σ=10） |
|--------|------|----------------------|
| **创新点1** 自适应噪声阈值 | 局部8×8窗口统计法替代全局标准差，避免多亮度区域噪声高估 | 0.050 → **0.653**（+1206%） |
| **创新点2** 方向自适应权重 | Softmax归一化方向权重，抑制低分辨率图像弱方向噪声干扰 | 0.653 → **0.712**（+9%） |
| **创新点3** 双阈值滞后二值化 | 基于PC响应百分位数的双阈值+连通性传播，替代Otsu | 0.712 → **0.799**（+12%） |

---

## 📊 实验结果 / Results

### 精度对比（无噪声）/ Accuracy Comparison (σ=0)

| 算法 | Precision | Recall | **F1** | 耗时(ms) |
|------|-----------|--------|--------|----------|
| 差比和（基线）| 79.7% | 95.1% | 0.867 | 0.09 |
| Sobel | 93.9% | 99.9% | 0.968 | 0.10 |
| Canny | 94.7% | 98.7% | 0.966 | 0.11 |
| **本文 SPC** | **95.0%** | **96.7%** | **0.958** | 2.40 |

### 鲁棒性测试 / Robustness Test

| 噪声强度 | 差比和 | Sobel | Canny | **SPC** | SPC vs Canny |
|----------|--------|-------|-------|---------|--------------|
| σ=0 | 0.867 | 0.968 | 0.966 | **0.958** | -0.8% |
| σ=5 | 0.804 | 0.689 | 0.959 | **0.956** | -0.3% |
| σ=10 | 0.533 | 0.321 | 0.334 | **0.799** | **+139.2%** |
| σ=15 | 0.363 | 0.271 | 0.209 | **0.478** | **+128.7%** |
| **平均** | 0.642 | 0.562 | 0.617 | **0.798** | **+29.3%** |

> 测试图像：MT9V034摄像头在真实智能车赛道环境下拍摄，188×120分辨率灰度图像。
> 真实传感器噪声水平 σ̂ ≈ 1.01，测试范围 [0, 5, 10, 15] 分别对应 0×/5×/10×/15× 真实噪声。

---

## 🗂️ 项目结构 / Repository Structure

```
SPC-Edge-Detection/
│
├── README.md                    # 本文件
│
├── python/                      # PC端仿真代码
│   ├── edge_detection_pc.py     # 主程序：四算法对比 + 鲁棒性测试 + 消融实验
│   ├── requirements.txt         # Python依赖
│   └── results/                 # 实验结果图输出目录
│       ├── result_comparison.png       # 四算法边缘对比图
│       ├── result_robustness.png       # 鲁棒性测试折线图
│       └── result_pc_internals.png     # SPC内部响应可视化
│
├── embedded/                    # CYT4BB7嵌入式C代码（定点化实现）
│   ├── spc_edge.h               # SPC模块头文件（含可调参数宏）
│   ├── spc_edge.c               # SPC算法全定点化实现
│   ├── camera.h                 # 摄像头模块头文件
│   └── camera.c                 # 集成SPC的摄像头处理模块
│
└── docs/                        # 文档资源
    ├── figures/                 # 论文图表（draw.io源文件）
    │   ├── fig2-1_hardware_system.drawio
    │   ├── fig3-1_spc_flowchart.drawio
    │   ├── fig3-2_noise_threshold.drawio
    │   └── fig3-3_dual_threshold.drawio
    └── paper/                   # 论文PDF（发表后更新）
```

---

## 🚀 快速开始 / Quick Start

### 环境要求 / Requirements

```bash
Python >= 3.8
numpy >= 1.24
opencv-python >= 4.8
scipy >= 1.10
matplotlib >= 3.7
```

### 安装依赖 / Install

```bash
git clone https://github.com/CloudeBamboo/SPC-Edge-Detection.git
cd SPC-Edge-Detection
pip install -r python/requirements.txt
```

### 运行仿真 / Run Simulation

```bash
cd python
python edge_detection_pc.py
```

运行后将在 `results/` 目录生成三张结果图：
- `result_comparison.png`：四种算法边缘检测结果对比
- `result_robustness.png`：不同噪声强度下各算法F1分数变化曲线
- `result_pc_internals.png`：SPC算法内部响应可视化（4方向热力图、噪声阈值直方图、方向权重柱状图）

### 自定义测试图像 / Custom Image

在 `edge_detection_pc.py` 顶部修改图像路径：

```python
# 修改此处为你的测试图像路径（建议使用纯英文路径）
IMAGE_PATH = r'D:\your\path\to\test_image.png'
```

---

## ⚙️ 可调参数 / Tunable Parameters

SPC算法的所有可调参数集中在 `embedded/spc_edge.h` 中：

```c
/* 高斯预处理开关：1=开启（推荐），0=关闭（更快） */
#define SPC_GAUSSIAN_ENABLE     1

/* 噪声阈值系数 k，T = k × σ̂
 * 越大：边缘越少越干净；越小：弱边缘保留越多
 * 建议范围：2 ~ 5，默认 3（对应3σ准则） */
#define SPC_NOISE_K             3

/* 强边缘百分位（0~100）
 * 越小：强边缘种子越多，边缘越粗
 * 越大：强边缘种子越少，边缘越细但可能断裂
 * 建议范围：30 ~ 60，默认 40 */
#define SPC_HIGH_PCTILE         40

/* 低阈值比例，T_low = T_high × ratio / 256
 * 对应 T_low = T_high × 0.2（ratio=51）
 * 建议范围：30 ~ 80 */
#define SPC_LOW_RATIO           51

/* 连通传播邻域半径（像素）
 * 越大：边缘越连续，速度略慢
 * 建议：1（3×3邻域）或 2（5×5邻域，默认） */
#define SPC_CONN_RADIUS         2
```

---

## 🔧 硬件平台 / Hardware Platform

| 组件 | 型号 | 规格 |
|------|------|------|
| 微控制器 | 英飞凌 CYT4BB7 | 双核 ARM Cortex-M7 @ 250 MHz，768 KB SRAM，4 MB Flash |
| 摄像头 | 逐飞科技 总钻风 | MT9V034，全局快门，188×120（4×4 Binning），最高500 FPS |
| 开发环境 | IAR EW for ARM | Version 9.40.1 |
| 调试器 | CMSIS-DAP | 逐飞科技配套 ARM 调试器 |
| 开源库 | 逐飞CYT4BB7开源库 | https://gitee.com/seekfree/CYT4BB7_Library |

---

## 📐 算法流程 / Algorithm Pipeline

```
输入：灰度图像（188×120）
    │
    ▼
① 高斯预处理（5×5，σ=1.2）
    │
    ▼
② 4方向差分滤波器对卷积
   方向：0°、45°、90°、135°
   计算偶/奇对称响应 Re,θ 和 Ro,θ
    │
    ▼
③ 计算各方向幅值 Aθ = √(Re,θ² + Ro,θ²)
   局部窗口法估计噪声 σ̂ → 自适应阈值 T = 3σ̂  【创新点1】
    │
    ▼
④ Softmax方向自适应权重 wθ                      【创新点2】
    │
    ▼
⑤ 加权PC响应：PC(x) = Σwθ·max(Aθ-T,0) / (ΣAθ+ε)
    │
    ▼
⑥ 双阈值滞后二值化 + 连通性传播                 【创新点3】
    │
    ▼
输出：二值化边缘掩膜（255=边缘，0=背景）
```

---

## 📝 引用 / Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{liu2026spc,
  title   = {A Simplified Phase Congruency Edge Detection Algorithm
             for Resource-Constrained Embedded Platforms},
  author  = {Liu, Junjie},
  journal = {[期刊名称]},
  year    = {2026},
  note    = {Under review}
}
```

---

## 📄 许可证 / License

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢 / Acknowledgment

感谢逐飞科技提供CYT4BB7开发板及开源驱动库支持。

Thanks to SeekFree Technology for providing the CYT4BB7 development board and open-source driver library.

---

<p align="center">
  <sub>© 2026 Liu Junjie · Anhui University of Information Engineering</sub>
</p>
