# 🏙️ RAICOM 2024 平安城市P1/P2/P3算法

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv8-Pose-orange?logo=ultralytics" alt="YOLO">
  <img src="https://img.shields.io/badge/Competition-RAICOM平安城市-red" alt="Competition">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

## 📖 项目简介

**Raicom2024** 是一套专为 **RAICOM（机器人开发者大赛）“平安城市”赛项** 打造的视觉算法参考库。

本项目主要面向**第一次参加该赛题的同学（尤其是缺少比赛经验的大一新生）**。仓库系统性地整理了往年省赛预选赛（P1、P2、P3）三个核心视觉任务的算法实现，旨在帮助新手快速了解赛题难度、任务形式以及主流的解题思路，为后续的算法优化和国赛冲刺打下坚实基础。

> 💡 **资料来源**：关于本赛项的完整任务说明及评分标准，请查阅《2024 RAICOM 机器人开发者大赛平安城市预选赛（省赛）- 培训 PPT》。

## 🎯 赛题任务拆解 (P1 / P2 / P3)

本仓库包含三个独立的视觉任务模块，涵盖了传统图像处理与深度学习目标检测/姿态估计技术：

### 🛤️ P1: 轨道线识别 (传统 CV)
- **任务目标**：识别画面中的轨道线，判断轨道的延伸方向，并输出具体指令（`straight`、`left`、`right`）及偏转角度。
- **技术路线**：基于 OpenCV 的图像预处理、边缘检测（Canny）、霍夫直线检测（HoughLines）或轮廓拟合，计算轨道线的斜率与消失点。
- **核心文件**：`src/p1_track_line.py`

### 👥 P2: 人群属性识别 (深度学习)
- **任务目标**：统计画面中的总人数，并精准识别出人群的衣着颜色属性（红色系、蓝色系、灰黑色系人数）。
- **技术路线**：采用 **YOLOv8-Pose** 进行人体姿态关键点检测，提取人体躯干区域的 ROI，随后结合 HSV 色彩空间分析或颜色分类器，判定衣着颜色归属。
- **核心文件**：`src/p2_crowd_identification.py`

### 🔥 P3: 楼宇火灾定位 (OCR + 区域逻辑)
- **任务目标**：识别画面中楼宇的名称（OCR），并根据楼层区域的划分，判断火焰所在的具体楼层。
- **技术路线**：使用 PaddleOCR 或 EasyOCR 提取楼宇文本信息；通过图像分割或网格划分定位火焰（红色/橙色高亮区域）的坐标，将其映射到对应的楼层逻辑区间。
- **核心文件**：`src/p3_building_fire.py` *(注：原始 P3 代码文件部分丢失，当前版本为基础功能实现，欢迎补充完善)*

## 📂 项目结构

```text
Raicom2024/
├── src/                          # 重构后的核心算法代码 (推荐运行此目录下的文件)
│   ├── p1_track_line.py          # P1: 轨道线识别
│   ├── p2_crowd_identification.py# P2: 人群属性识别
│   └── p3_building_fire.py       # P3: 楼宇火灾定位
├── original_latest/              # 原始最新脚本备份 (便于对照与回溯)
├── data/                         # 测试数据集 (需自行准备，已 gitignore)
│   ├── P1/
│   ├── P2/
│   └── P3/
├── models/                       # 模型权重文件 (需自行下载，已 gitignore)
├── outputs/                      # 算法运行结果输出目录
├── docs/                         # 补充说明文档与培训 PPT
├── requirements.txt              # Python 依赖清单
└── README.md                     # 项目说明文档
