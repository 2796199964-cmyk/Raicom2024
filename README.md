# 2024 RAICOM 平安城市 P1/P2/P3 算法

本项目主要面向第一次参加 RAICOM 平安城市赛题的同学，尤其是缺少比赛经验的大一同学。仓库整理了往年省赛 P1、P2、P3 三个视觉任务的算法实现，可作为赛题难度、任务形式和解题思路的参考。

本仓库整理自 `2024RAICOM机器人开发者大赛平安城市预选赛（省赛）-培训PPT.pdf` 对应的三个视觉任务：

- `P1`：轨道线识别，判断轨道方向并输出 `straight`、`left`、`right` 及角度。
- `P2`：人群识别，基于 YOLO 姿态关键点统计总人数，并识别红色系、蓝色系、灰黑色系人数。
- `P3`：楼宇火灾识别，使用 OCR 识别楼宇名称，并根据楼层区域判断火焰所在楼层。

## 文件来源

打包时按各任务文件夹中“修改时间最新”的 `.py` 文件作为算法来源：

| 任务 | 原始最新文件 | 打包后的 GitHub 版 |
| --- | --- | --- |
| P1 | `P1/P1 - 副本.py` | `src/p1_track_line.py` |
| P2 | `P2/P2多角度pose.py` | `src/p2_crowd_identification.py` |
| P3 | `P3/P3.py` | `src/p3_building_fire.py` |

`original_latest/` 中保留了三份原始最新脚本，便于对照。

P3代码文件丢失，代码仅为部分功能

## 安装依赖

```bash
pip install -r requirements.txt
```

P2 需要 YOLO 权重文件。建议将权重放到 `models/` 目录，例如：

```text
models/yolov8l-pose.pt
```

由于模型权重和数据集通常体积较大，`.gitignore` 已默认排除 `*.pt`、数据集压缩包、图片数据和输出结果。上传 GitHub 时建议只上传算法代码与说明文档。

## 运行示例

```bash
python src/p1_track_line.py --input data/P1 --output outputs/P1 --sample-size 10
```

```bash
set P2_MODEL_PATH=models/yolov8l-pose.pt
set P2_INPUT_DIR=data/P2
set P2_OUTPUT_DIR=outputs/P2
python src/p2_crowd_identification.py
```

```bash
set P3_INPUT_DIR=data/P3
set P3_OUTPUT_DIR=outputs/P3
python src/p3_building_fire.py
```

## 目录建议

```text
data/
  P1/
  P2/
  P3/
models/
outputs/
src/
original_latest/
```

## 说明

- P1 已改为命令行参数配置输入、输出目录。
- P2 已改为通过环境变量配置模型、输入、输出路径。
- P3 已改为通过环境变量配置输入、输出路径，并为未识别到楼宇名称的情况提供默认值。
- 算法主体逻辑仍以打包时各 P 文件夹中最新 `.py` 为准。

打包时间：2026-06-22 01:12:26
