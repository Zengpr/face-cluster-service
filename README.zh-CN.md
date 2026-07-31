# Face Cluster Service — 中文详细说明

> 容器化的人脸聚类 API：`InsightFace ArcFace 特征提取 + 阈值聚类`，基于 FastAPI + Docker 部署。
> 为「鸿兴印刷 AI 科学面试 take-home」而构建，同时沉淀了生产级工程实践。

---

## 一、项目简介

**一句话：** 上传 N 张人脸照片，服务返回「这些人属于哪几组/哪几个人」的结果。

**核心思路：**
1. 用深度神经网络（ArcFace R50，`buffalo_l` 模型包）把每张人脸图片编码成 **512 维特征向量**（embedding）。
2. 同一张脸在特征空间里距离近，不同的人距离远。
3. 根据余弦相似度做聚类——相似度超过阈值的图片归为一组。

**本质是「以图识人」的最小闭环**：检测人脸 → 提取特征 → 特征聚类 → 分组输出。
可用于相册自动归类、监控视频人脸去重、证件照库按人分组等场景。

---

## 二、功能清单

| 类别 | 功能 | 说明 |
|------|------|------|
| **核心** | 同步聚类 `/cluster` | 上传图片，同步返回分组 |
| **核心** | 异步聚类 `/cluster/async` | 提交任务，返回 `task_id`，Redis 存结果 |
| **核心** | 任务状态轮询 | `GET /cluster/async/{task_id}` |
| **核心** | 双聚类后端 | `agglomerative`（单链连通分量，默认）/ `dbscan`（密度聚类） |
| **可观测** | `/health` | 存活探针（含 Redis ping、模型加载状态） |
| **可观测** | `/ready` | 就绪探针（确认 ONNX 模型真正加载） |
| **可观测** | `/metrics` | Prometheus 文本暴露格式 |
| **可观测** | 结构化 JSON 日志 | structlog，全链路带请求 ID |
| **生产化** | 限流 | 滑动窗口令牌桶，默认 60 req/min/IP，超限返回 429 |
| **生产化** | 请求追踪 | `X-Request-ID` 中间件，全链路透传 |
| **生产化** | 优雅关闭 | SIGTERM/SIGINT 后拒绝新请求（503），排空在途请求 |
| **生产化** | 硬性资源上限 | 最多 64 张/请求、单图 15 MiB、白名单 Content-Type |
| **生产化** | CORS | 可配置跨域白名单 |
| **演示** | Demo 模式 | `X-Demo-Mode: true` 头 → 结构化 stub 嵌入，无需真实人脸即可跑通全链路 |
| **扩展** | MCP Server | 面向 Agent 生态的工具接口（`cluster_faces`） |
| **扩展** | 精度评估脚本 | NMI / ARI / V-Measure / 同质性与完整性 |
| **工程** | CI | GitHub Actions：单测 + API 测试 + Docker 构建冒烟 |
| **工程** | 压测 | JMeter 5.6 场景 + Python 压测脚本 + HTML 报告 |

---

## 三、核心流程（端到端）

```
图片字节 → 解码(OpenCV) → 人脸检测(RetinaFace) → 特征提取(ArcFace 512-d)
        → 余弦相似度矩阵 → 单链聚类(union-find) → 分组输出 + 轮廓系数
```

**POST /cluster 请求生命周期：**

1. **FastAPI 校验** —— 校验 `files` 表单、Content-Type 白名单、图片预算（空/超 64 张拒绝）。
2. **强制加载模型** —— `embedder_loaded` 依赖确保模型惰性加载一次。
3. **CPU 密集计算卸载** —— `asyncio.to_thread(run_cluster, ...)` 把阻塞的推理放到线程池，
   uvicorn 事件循环保持响应。
4. **解码** —— `cv2.imdecode` 直接在内存字节上解码（零磁盘 I/O），BGR→RGB 转换一次。
5. **特征提取** —— 对每张图取**最大人脸**，返回 L2 归一化的 512 维向量。
   检测不到人脸的图片被丢弃，并在 `dropped_files` 中报告。
6. **聚类** —— 计算 N×N 余弦相似度矩阵，在给定阈值下跑**单链 union-find**；
   形成 ≥2 组时计算轮廓系数（silhouette），用于离线调阈值。
7. **序列化** —— pydantic v2 → JSON 返回。

**异步路径：** 提交任务 → 返回 `task_id` → 后台 worker 跑同样的 pipeline →
结果写入 Redis `task:result:{task_id}`（TTL 1 小时）→ 前端轮询。

---

## 四、架构设计

### 4.1 分层结构

```
┌─────────────────────────────────────────────────────┐
│  展示层  FastAPI + uvicorn（同步/异步路由，中间件）     │
│    ├─ RequestIDMiddleware  请求追踪                    │
│    ├─ 限流中间件（/health /ready /metrics 豁免）        │
│    └─ 优雅关闭守卫（503 拒绝新请求）                    │
├─────────────────────────────────────────────────────┤
│  业务层  app/services                                 │
│    ├─ preprocess.decode_image   图片解码               │
│    ├─ FaceEmbedder（单例）       特征提取 + stub 降级   │
│    ├─ clusterer                 双后端聚类引擎         │
│    └─ pipeline.run_cluster      编排（字节→分组 DTO）   │
├─────────────────────────────────────────────────────┤
│  领域层  app/core                                     │
│    ├─ config（pydantic-settings 环境变量注入）          │
│    ├─ errors（错误码体系 4001-5005）                   │
│    ├─ logging（structlog 结构化日志）                   │
│    └─ metrics（Prometheus 文本暴露）                    │
├─────────────────────────────────────────────────────┤
│  数据层  Redis（异步任务状态/结果存储，TTL 1h）          │
└─────────────────────────────────────────────────────┘
```

### 4.2 目录结构

```
face-cluster-service/
├── app/
│   ├── api/              # FastAPI 路由（cluster、meta、deps）
│   ├── core/             # config、errors、logging、metrics、rate_limit、middleware
│   ├── models/           # pydantic schemas
│   └── services/         # face_embedder、clusterer、pipeline、preprocess、tasks
├── tests/                # 单元测试 + API 测试 + 集成测试
├── scripts/              # 模型下载、demo 测试、精度评估、压测、MCP 服务器
├── jmeter/               # JMeter 压测计划 + 报告
├── .github/workflows/    # CI（pytest + docker build）
├── Dockerfile            # 多阶段构建：builder → runtime
├── docker-compose.yml    # app + redis
└── docs/                 # 架构/测试/性能/面试笔记
```

### 4.3 为什么这样选型

| 决策 | 选择 | 理由 |
|------|------|------|
| Web 框架 | **FastAPI** | 原生 async、pydantic v2、自动 OpenAPI 文档、`Depends()` 依赖注入 |
| 人脸特征 | **InsightFace buffalo_l** | 业界标准的 ArcFace 512-d、ONNX 免费可移植、CPU/GPU 双跑 |
| 聚类算法 | **单链阈值聚类** | 人脸聚类参考配方——传递闭包式分组，跨多张照片识别同一身份 |
| 异步任务 | **Redis + asyncio.to_thread** | 演示无需 Celery；CPU 密集必须释放事件循环；Redis 支撑多 worker 共享状态 |
| 容器初始化 | **tini + 入口脚本** | 防僵尸进程；确保 ONNX 权重在 uvicorn 绑定前就绪；下载失败优雅降级 |
| 可观测性 | **Prometheus 暴露格式** | 业界默认，开箱即配 Grafana；counter/histogram 对齐 USE/RED 规范 |

---

## 五、技术细节

### 5.1 人脸特征提取（`face_embedder.py`）

- **单例模式** + 线程锁：模型在 worker 生命周期内常驻，避免每次请求 ~0.5s 冷启动。
- **三级降级兜底：**
  1. 真实模型加载成功 → 用 ONNX 推理；
  2. 模型包缺失（防火墙环境）→ 自动转 **stub 模式**，基于图片内容 MD5 生成确定性伪特征；
  3. **Demo 模式**（`X-Demo-Mode: true` 头 / 环境变量）→ 结构化伪特征：
     图片按内容哈希归入 3 个身份质心，聚类结果干净可复现，无需真实人脸。
- 取**最大人脸**做主特征，`embed_image` 返回 `(512-d 归一化向量, 检测到的人脸数)`。
- 推理失败抛 `InferenceError`；无人脸抛 `NoFaceError`。

### 5.2 聚类引擎（`clusterer.py`）

**后端一：agglomerative（默认，单链连通分量）**
- 建 N×N 余弦相似度矩阵（L2 归一化后内积）。
- **并查集（union-find）**：两两相似度 ≥ 阈值则合并。带路径压缩。
- 传递闭包特性：A≈B、B≈C → A、B、C 同一组（适合同人跨多张照片）。

**后端二：DBSCAN（密度聚类）**
- 阈值视为邻域半径，`min_samples` 控制核心点。
- 邻居数不足的点标记为 `-1`（噪声），天然支持离群点剔除。

**统一输出 `ClusterResult`：**
- `labels`（压缩到 0..K-1，噪声保持 -1）、`n_clusters`、`cluster_sizes`、`silhouette`。
- 双后端同一数据结构，API 层可透明切换。

**轮廓系数：** 聚类 ≥2 组时用 sklearn 计算，为离线调阈值提供量化依据。

### 5.3 编排层（`pipeline.py`）

`run_cluster`：字节流 → 逐图解码/嵌入/收集 → 矩阵 → 聚类 → 组装 DTO。
- 保证输出确定性：每个分组内文件名排序，分组按 ID 排序。
- `dropped_files` 报告解码失败或无人脸的图片。

### 5.4 生产化组件

| 组件 | 实现 | 关键点 |
|------|------|--------|
| **限流** `rate_limit.py` | 滑动窗口令牌桶，`defaultdict[ip] → timestamps` | `/health`、`/ready`、`/metrics` 豁免；429 带 `Retry-After: 60` |
| **请求追踪** `middleware.py` | `X-Request-ID` 透传/生成 | 无则生成 UUID，响应头回写 |
| **优雅关闭** | SIGTERM/SIGINT 置 `_shutting_down` → 503 守卫 | 拒绝新请求，在途请求正常完成 |
| **错误体系** `errors.py` | `ErrCode` 枚举（4001-5005）+ `ServiceError` | 统一 `{detail:{error:{code,name,message}}}` 形状 |

### 5.5 MCP Server（`scripts/mcp_server.py`）

基于 FastMCP，暴露 `cluster_faces` 工具——让 Claude 等 Agent 直接调用人脸聚类能力，
支持 stdio 传输 + 无 FastMCP 环境的降级实现。**服务人类 API 之外的 Agent 生态入口。**

### 5.6 精度评估（`scripts/evaluate.py`）

生成带真实身份标签的基准数据 → 聚类 → 用 sklearn 算：
**ARI**（兰德指数）、**NMI**（归一化互信息）、**Homogeneity**、**Completeness**、**V-Measure**。

---

## 六、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 存活 + Redis ping + 模型加载状态 |
| GET | `/ready` | 就绪（确认 onnx 加载） |
| POST | `/cluster` | 同步聚类 |
| POST | `/cluster/async` | 异步提交，返回 `task_id` |
| GET | `/cluster/async/{task_id}` | 轮询任务状态 |
| GET | `/metrics` | Prometheus 文本暴露 |

### 示例：POST /cluster

```bash
curl -X POST http://localhost:8000/cluster \
  -F "files=@a.png" -F "files=@b.png" -F "files=@c.png" \
  -F "threshold=0.6" \
  -F "backend=agglomerative" \
  -H "X-Demo-Mode: true"      # 演示模式：无需真实人脸
```

响应：

```json
{
  "ok": true,
  "n_images": 9,
  "n_clusters": 3,
  "n_noise": 0,
  "threshold": 0.6,
  "backend": "agglomerative",
  "silhouette": 0.41,
  "cluster_sizes": {"0": 3, "1": 3, "2": 3},
  "clusters": [
    {"cluster_id": 0, "files": ["alice_1.png", "alice_2.png", "alice_3.png"]},
    {"cluster_id": 1, "files": ["bob_1.png",   "bob_2.png",   "bob_3.png"]},
    {"cluster_id": 2, "files": ["eve_1.png",   "eve_2.png",   "eve_3.png"]}
  ],
  "label_by_file": {"alice_1.png": 0, "bob_1.png": 1, "eve_1.png": 2},
  "dropped_files": []
}
```

### 错误模型

所有非 2xx 响应统一形状：

```json
{ "detail": { "error": { "code": 4006, "name": "BAD_FILE_PAYLOAD",
                          "message": "Could not decode image" } } }
```

完整错误码表见 `docs/TESTING.md`（涵盖 4001-5005，如 `NO_IMAGES`、`NO_FACE`、
`TOO_MANY_IMAGES`、`MODEL_LOAD`、`INFERENCE` 等）。

---

## 七、配置项（`app/core/config.py`，env 可覆盖）

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `DETECTOR_NAME` | `buffalo_l` | InsightFace 模型包名 |
| `DEFAULT_THRESHOLD` | `0.6` | 余弦相似度阈值 |
| `CLUSTERING_BACKEND` | `agglomerative` | `agglomerative` / `dbscan` |
| `MIN_SAMPLES_FOR_CLUSTER` | `2` | DBSCAN 核心点数 |
| `MAX_IMAGES_PER_REQUEST` | `64` | 单请求硬上限 |
| `MAX_IMAGE_BYTES` | `15 MiB` | 单图大小上限 |
| `REDIS_URL` | `redis://redis:6379/0` | 异步结果存储 |
| `RATE_LIMIT_PER_MINUTE` | `60` | 每 IP 每分钟限流 |
| `DEMO_MODE` | `false` | 全局 demo 模式 |
| `CTX_ID` | `-1` | -1=CPU；≥0 时切 CUDA |
| `DET_SIZE` | `640` | 检测分辨率 |

---

## 八、部署

```bash
git clone https://github.com/Zengpr/face-cluster-service.git
cd face-cluster-service
docker compose up -d --build
curl http://localhost:8000/health
```

- **多阶段 Dockerfile**：builder（pip 装到 `/install`）→ runtime（slim + tini）。
- **镜像瘦身**：OpenCV-headless（无 GUI 依赖）、ONNX Runtime CPU 版。
- **冷启动**：首次 ~1-2 分钟下载 `buffalo_l`（~300MB，docker volume 缓存）。
- **Compose**：`app(8765:8000)` + `redis:6-alpine(6380:6379)`，均带 healthcheck。
- **无模型时优雅降级**：`download_model.sh` 失败 → stub 模式，服务照常启动。
- **CI**：GitHub Actions 跑 pytest + Docker 构建冒烟。

---

## 九、测试与性能

### 测试策略（`tests/`）

| 层级 | 内容 |
|------|------|
| 单元 | `test_clusterer.py`（纯 numpy，双后端、噪声、边界） |
| API | `test_api.py`（TestClient + stub 嵌入，错误码、限流、预算） |
| 集成 | `test_smoke_live.py`（真实 HTTP 冒烟） |
| 演示 | `scripts/test_demo.py`（合成图 + `X-Demo-Mode`，端到端验证） |

### 实测压测结果（`scripts/load_test.py`，4 线程 × 20 请求，容器内）

```
Wall time:      0.2s
OK:             20（0 错误）
Throughput:     88.8 req/s
p50 latency:    39 ms
p90 latency:    52 ms
p99 latency:    57 ms
```

### 性能设计要点

- **CPU 密集推理**（单图 ~0.3-1s）通过 `asyncio.to_thread` 卸载，不阻塞事件循环。
- **N×N 聚类 O(N²)**：请求上限 64 张；更大规模建议向量库（FAISS/Milvus/pgvector）。
- **模型常驻**：单例加载，避免每请求重载。
- 基准见 `docs/PERFORMANCE.md`（JMeter 方法论 + 解读指南）。

---

## 十、横向扩展与限制

```
        ┌─────────── Load Balancer（Nginx / Traefik / ALB）───────────┐
        │                                                             │
  ┌─────┴────┐ ┌─────┴────┐ ┌─────┴────┐
  │ pod-1    │ │ pod-2    │ │ pod-N    │  每个 pod 4-8 个 uvicorn worker
  └──────────┘ └──────────┘ └──────────┘
        └──────────── 共享：Redis（HA）、模型缓存 PVC/NFS ────────────┘
```

- **同步路径完全无状态**：相同图片 → 确定性嵌入与聚类，任意 pod 可服务任意请求。
- **异步路径升级路径**：`asyncio.create_task` → Arq worker（Redis 队列），worker 池横向扩展。
- **限制与缓解：**
  - CPU 密集 → 上游抽嵌入入向量库复用；
  - N² 聚类 → 超 64 张走离线批量作业（IVF/HNSW）；
  - 模型冷启动 → PVC/S3 预置缓存卷；
  - Redis 单点 → Cluster/Valkey，API 可降级内存存储。

### 安全设计

- 仅接受 `image/jpeg|png|webp|bmp` Content-Type。
- 硬上限：64 张 & 15 MiB/图。
- 零磁盘写入：全流程 `numpy.frombuffer` 内存流式处理（防路径穿越）。
- 生产建议：API key 鉴权、`slowapi` 分布式限流、S3 签名上传代替 multipart。

---

## 十一、面试亮点（中文版）

1. **真实算法闭环**：不是玩具——RetinaFace 检测 + ArcFace 512-d 特征 + 单链传递闭包聚类，
   全部是生产人脸聚类管线的标准组件。
2. **工程素养**：错误码体系、结构化日志、Prometheus 指标、请求 ID 全链路追踪、
   限流、优雅关闭、CORS——每个都是真实线上服务必备。
3. **优雅降级**：模型加载失败不崩服，自动转 stub 模式——「离线环境也能全链路测试」，
   这是交付工程里极有说服力的设计。
4. **可演示**：`X-Demo-Mode: true` 让面试官 30 秒内看到聚类效果，无需真实人脸数据。
5. **可量化**：压测报告（p50=39ms）、精度评估（NMI/ARI）、JMeter 计划，全部有据可查。
6. **面向未来**：MCP Server 让服务接入 Agent 生态；Arq 队列给出异步横向扩展路线图。
