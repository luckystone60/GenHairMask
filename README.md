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
fine_hair_mask_01.png          严格 0/1 的 uint8 细发丝 mask（主结果）
fine_hair_mask.png             兼容普通看图软件的 0/255 二值 mask
fine_hair_alpha_16bit.png      BiRefNet 原始覆盖率，仅保留最终细发丝
fine_hair_score_16bit.png      16-bit 融合置信度
fine_hair_bok_blend.png        01 mask 与完整 BOK 的红色混合验收图
fine_hair_overlay.jpg          JPEG 兼容预览
validation_report.json         尺寸、位深、二值性自动检查
debug/00_*.jpg、01_*.png...    按 `00_`～`27_` 处理步骤编号的 ROI 尺寸诊断图
```

算法组合：Sapiens2 Hair 语义邻域 → 按图像分辨率和 Hair 主体尺度生成自适应常规搜索区 → 排除 Face/Apparel/Clothing 类边界 → BiRefNet alpha 边缘 → BOK 多尺度中值残差 → EDOF/BOK 清晰度负证据 → 双阈值连通 → 只保留约 12 像素 Hair 内侧边缘带 → 形态学宽结构剔除 → 四方向高置信短缺口连接 → 颜色和细长结构约束的远距离延伸区生长。

默认使用 `--search-mode adaptive`。常规搜索半径取“4K 参考半径”和“Hair 主体最长边比例”中的较大值并设置分辨率相关上限；远距离延伸半径再由区域生长预设决定。超出常规搜索区后会自动提高颜色、方向一致性、细线、alpha/Hair 证据和虚化负证据门槛，因此不是在整张图上无约束生长。区域生长把生长前已确认的发丝作为固定颜色锚点，不会把新增像素继续当作颜色基准而产生颜色漂移。

区域生长提供四档参数：

```powershell
--growth-preset off             # 完全关闭区域生长
--growth-preset conservative    # 背景复杂、优先精度
--growth-preset balanced        # 默认，召回/精度平衡
--growth-preset recall          # 发丝漏检较多、优先召回
```

在固定搜索模式的 2p 样例上，`balanced` 从生长前的 36,444 像素增加到 42,074（+15.4%）；`recall` 增加到 48,644（+33.5%）。2p 没有明显超出旧搜索区的长发丝，因此自适应模式主要用于全量长发数据，不应只依据 2p 像素数判断收益。高召回档应检查 blending、`debug/21_region_growth_added.png` 和 `debug/25_growth_extension_added.png`。

全量数据仍有较长发丝被截断时，推荐先使用：

```powershell
--growth-preset recall `
--search-mode adaptive `
--search-radius-scale 0.24 `
--search-max-radius 480
```

背景误检增加时，优先退回 `balanced/conservative`，或降低 `--search-radius-scale`，不要取消搜索区硬上限。`--growth-radius-scale`、`--growth-max-radius`、`--growth-color-delta`、`--growth-line-min` 和 `--growth-coherence-min` 均可逐项覆盖预设。

需要恢复旧版更保守的输出时：

```powershell
python extract_fine_hair.py `
  --prefix D:\images\base\2p `
  --output D:\results\2p-conservative `
  --thin-radius 6 `
  --gap-close-radius 0 `
  --search-mode fixed `
  --growth-preset off
```

背景复杂时优先缩短桥接距离或提高桥接分数，不建议直接全局降低双阈值：

```text
--gap-close-radius 2 --gap-score-min 0.20
```

## 单图 prefix 模式

只需给出公共路径前缀，脚本会自动寻找其余输入：

```powershell
python extract_fine_hair.py `
  --prefix D:\images\base\2p `
  --output D:\results\2p
```

至少需要：

```text
2p_bok.png
2p_edof.png
2p_biref_bok_mat4k.png    # 也兼容 2p_bok_mat4k.png
```

可选先验：

```text
2p_bok_hair.png
2p_bok_hair_probability_16bit.png
2p_bok_sapiens2_labels.png
```

若 Hair mask 缺失或为空，脚本会用 matte 缩小 ROI，并优先使用剩余的 Sapiens2 Hair 概率/标签；所有 Hair 先验都缺失时才把 matte 当作种子。此模式保证不中断，但只能检测人像轮廓附近的细线，误检风险高于正常 Hair 模式，具体后备来源会写入 `run_metadata.json`。

旧版显式参数 `--bok --edof --hair --hair-probability --sapiens2-labels --matte` 仍可使用，其中 `--hair` 已变为可选。

## 目录批处理

批处理会扫描目录内所有 `<prefix>_bok` 图片，按 prefix 串行处理，并把结果分别写入独立子目录：

```powershell
python extract_fine_hair.py `
  --input-dir D:\images\base `
  --batch `
  --output D:\results\fine-hair
```

汇总结果位于 `batch_summary.json`。缺 Hair mask 会自动 fallback；缺少 EDOF 或 matte 的样本会记录失败，但默认继续处理其他样本。加 `--fail-fast` 可在首个错误处停止。

## ROI 加速

默认根据 Hair mask（缺失时根据 matte）计算外接框，并自动增加“有效常规搜索半径 + 有效远距离延伸半径 + filter halo”，只在 crop 中运行中值滤波、梯度和形态学操作。最终 01 mask、alpha 和 score 均回填到原始完整分辨率且 ROI 外严格为 0；BOK blending 在 ROI 外保持原始 BOK。可用 `--no-roi` 关闭裁剪；`--roi-margin` 可设置外扩下限，但不会低于保证滤波等价性的安全值。

## 将细发丝背景化

`hair_bokeh.py` 使用最终细发丝 mask，把 BOK 中对应的发丝区域替换为邻近背景。默认不是直接中值滤波，而是：mask 轻微外扩以覆盖抗锯齿边缘 → Telea 重建原有 BOK 背景 → 对重建区域做轻微高斯虚化 → 在线性光空间羽化融合。由于背景样本直接来自 BOK，原图已有的散景颜色和模糊形态会被保留下来。

```powershell
python hair_bokeh.py `
  --prefix D:\images\base\2p `
  --fine-dir D:\results\2p\final `
  --output D:\results\2p\hair-bokeh
```

也可以显式指定输入：

```powershell
python hair_bokeh.py `
  --bok D:\images\2p_bok.png `
  --mask D:\results\fine_hair_mask_01.png `
  --alpha D:\results\fine_hair_alpha_16bit.png `
  --output D:\results\hair-bokeh
```

主要输出：

```text
hair_bokeh.png                 完整分辨率主结果
hair_bokeh_compare.jpg         BOK 与处理结果左右对比
hair_bokeh_metadata.json       输入、ROI、参数与像素统计
debug/00_* ～ 06_*             重建区域、融合权重和差值诊断图
```

`--method median` 可用于快速对照，但人物轮廓附近可能把主体颜色带入背景；`--method hybrid` 会给 Telea 结果混入少量中值结果。默认 `--method inpaint` 对 2p 样例最稳妥。残留亮边可提高 `--expand-radius` 到 3；边缘过软可把 `--feather-sigma` 降到 1.5；背景本身非常虚时可提高 `--background-blur-sigma`。

## 许可

BiRefNet 为 MIT。Sapiens2 使用 Meta 的 Sapiens2 License；商用前必须自行确认上游许可条款。
