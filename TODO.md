# TODO

## 人脸聚类

- [ ] **加时间轨迹投票（对侧脸最有效）**
  - 现状：`faces/cluster_faces.py` 每 1s 抽帧后逐帧独立匹配/聚类；`person_00` 成了磁铁簇，混进多人的侧脸和低质量帧。
  - 做法：同一镜头内按人脸框 IoU（≥ 0.25）且时间间隔 ≤ 2s 串成 track；track 内 embedding 求均值（或取 medoid）后再匹配/聚类；换镜头时 track 自然断开，避免跨场景串人。
  - 前置：需要把 embedding 落盘（如 `embeddings.npy`），否则无法离线对 track 聚合重算。
  - 状态：暂缓，先不改代码。

## 台词一致性判别模型（双图）

- [ ] **训练一个双图输入模型：判断两张画面的台词是否相同**
  - 任务：输入同一部片的相邻两帧（或任意两帧），输出「字幕文本是否一样」，用于替代/校验 VLM 逐帧提取后的去重合并（`SRT_MERGE_SIMILARITY`、`make_srt.py` 目前靠文本相似度后处理）。
  - 数据：现有 `data_making/srt_model/` 的 SFT 数据（图 + 字幕文本）可作为正负样本来源；正样本取字幕文本一致的两帧，负样本取不一致的两帧，需注意两帧画面本身可能不同（镜头切换）带来的干扰。
  - 待定：backbone（复用 `classifier/model.py` 的 mobilenet_v3 / resnet18）、双图融合方式（concat / 特征差 / cross-attention）、输出用二分类还是相似度、是否直接预测字幕文本。
  - 状态：未开始，仅记录需求。

