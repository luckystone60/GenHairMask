# 4K 精细边缘发丝 Mask 工程

本工程以 BOK 图为唯一坐标系，生成四张同分辨率基础图，并提取**细碎、边缘、飘散、单根**发丝；不会把大片头发主体合并进最终 mask。

## 已选模型

- `ZhengPeng7/BiRefNet_HR-matting`：2048 输入的高分辨率通用 matting，输出 16-bit 人像 alpha。
- `facebook/sapiens2-seg-0.4b`：Sapiens2 29 类人体部件分割；Hair 是类别 4。选择 0.4B 是为了在 RTX 2060 SUPER 8GB 上稳定 FP16 推理。

模型 revision 已固定在 `run_pipeline.py`，避免远端代码或权重更新导致结果漂移。首次运行约下载 1.93 GiB；后续自动离线复用。

## 一键运行

```powershell
git clone https://github.com/luckystone60/GenHairMask.git
cd GenHairMask
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

准备同一前缀的两张输入图，例如：

```text
sample/2p_bok.jpg
sample/2p_edof.jpg
```

默认读取项目下的 `sample/`，输出到 `results/2p/`：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sample.ps1
```

也可以指定输入目录、文件前缀与输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sample.ps1 `
  -Sample D:\images\pair01 `
  -Prefix pair01 `
  -Output D:\hairmask-results\pair01
```

输入照片、模型权重、虚拟环境和推理结果不会提交到仓库。模型权重由 Hugging Face 在首次运行时下载到 `models/`。

## 四张基础图

均为 `2509 × 3760`，位于 `results/2p/base/`：

```text
2p_bok.png          BOK 原生坐标、JPEG 解码后无损 PNG
2p_edof.png         EDOF 缩放至 BOK 尺寸
2p_bok_hair.png     Sapiens2 粗 Hair 二值 mask
2p_bok_mat4k.png    BiRefNet 16-bit 人像 alpha
```

辅助先验：

```text
2p_bok_hair_probability_16bit.png
2p_bok_sapiens2_labels.png
```

原 EDOF 是 `3191 × 4780`，BOK 是 `2509 × 3760`。SIFT/RANSAC 只用于诊断；可靠匹配的中位偏移不足 1 像素，因此 canonical EDOF 采用 resize-only，避免用 AIGC 已改变的局部区域拟合仿射后扭曲其他区域。EDOF/BOK 的局部比较使用 9×9 邻域梯度，对几像素非一致具有容忍度。

## 最终输出

位于 `results/2p/final/`：

```text
fine_hair_mask.png             4K 二值细发丝 mask（主结果）
fine_hair_alpha_16bit.png      BiRefNet 原始覆盖率，仅保留最终细发丝
fine_hair_score_16bit.png      融合置信度
fine_hair_overlay.jpg          红色叠加验收图
validation_report.json         尺寸、位深、二值性自动检查
debug_*.png                    各阶段诊断图
```

算法组合：Sapiens2 Hair 语义邻域 → 排除 Face/Apparel/Clothing 类边界 → BiRefNet alpha 边缘 → BOK 多尺度中值残差 → EDOF/BOK 清晰度负证据 → 双阈值连通 → 只保留约 12 像素 Hair 内侧边缘带 → 形态学宽结构剔除。

默认参数已经针对 2p 样例调好。更保守可提高 `--low-score` / `--high-score`；漏发丝可增大 `--inner-band` 或 `--thin-radius`。`thin-radius` 越大，允许保留的发丝越粗。

## 许可

BiRefNet 为 MIT。Sapiens2 使用 Meta 的 Sapiens2 License；商用前必须自行确认上游许可条款。
