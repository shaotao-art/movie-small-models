# TODO

## 人脸聚类

- [ ] **加时间轨迹投票（对侧脸最有效）**
  - 现状：`cluster_faces.py` 每 1s 抽帧后逐帧独立匹配/聚类；`person_00` 成了磁铁簇，混进多人的侧脸和低质量帧。
  - 做法：同一镜头内按人脸框 IoU（≥ 0.25）且时间间隔 ≤ 2s 串成 track；track 内 embedding 求均值（或取 medoid）后再匹配/聚类；换镜头时 track 自然断开，避免跨场景串人。
  - 前置：需要把 embedding 落盘（如 `embeddings.npy`），否则无法离线对 track 聚合重算。
  - 状态：暂缓，先不改代码。
