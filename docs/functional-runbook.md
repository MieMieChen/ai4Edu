# 本机真实评分与模型评测

当前使用真实 PostgreSQL、Redis、Celery、FastAPI 和百炼接口。数据库及 Redis 在 Docker 中，应用进程使用项目 `.venv`。输入是主观题，评分细则自动生成及核验，没有教师审核步骤。

本手册记录当前工作区的配置。原始 ZIP、整个 `data/` 目录和凭据均不在代码仓库中；其他机器需先准备自己的凭据和输入，并按 README 启动依赖。

当前开发配置只为`grade`启用4096 Token思考；正常已转正的按题输入关闭方向模型，直接转写和评分。整卷仍需题目定位。下面的C策略命令用于复现历史A/B/C对照，当前开发批次使用`--strategy B`。正常图回归记录在本机`data/runs/normal-thinking-v1/`，尚未通过独立质量验收。

## 当前机器的启动方式

已有其他项目使用 6379，因此本项目数据库端口为 15432、Redis 为 16379，API 为 18000。`.env` 已配置这些连接，密钥继续通过 `bailian.txt` 读取。

```bash
# 首次或 Colima 停止时；不改变已有 Docker context 与 SSH 配置
colima start --activate=false --ssh-config=false

# 当前工作区已下载并校验官方 Compose v5.6.0，保存在忽略目录
./data/tools/docker-compose -p ai4edu -f compose.yaml -f data/runtime/ports.yaml up -d postgres redis

# 显式初始化并升级新增字段；保留已有任务和评分结果
.venv/bin/ai4edu init-db

# 前台运行 API、准备/评分/调度 Worker 和 Beat；Ctrl-C 一起停止应用进程
.venv/bin/python scripts/serve_local.py
```

评分工作台：<http://127.0.0.1:18000/demo/>，API 文档：<http://127.0.0.1:18000/docs>。日志在 `data/runtime/{api,preparation,grading,control,scheduler}.log`；`services.json` 记录本次 supervisor 与子进程 PID。依赖容器可用同一 Compose 命令的 `stop postgres redis` 停止，不删除数据卷。

本机 `.env` 的 `AI4EDU_EVALUATION_LABELS_FILE=data/evaluation/2026-11-07/labels.jsonl` 为工作台启用独立参考分对照。仅对实际对应这些样例的考生编号与题号使用此文件，其他试卷需改用自己的标签。标签未配置或不能完整匹配批次时，不展示一致率。浏览已有批次、证据和离线评估不会新增模型调用。

标准全容器部署仍使用根目录 `compose.yaml` 和 README 的命令。本轮真实验证采用上面的依赖容器与本机应用组合，不能把它写成完整应用镜像的运行验证。

## 数据与冻结

```bash
.venv/bin/ai4edu prepare-eval --archive 测试数据集.zip --output data/evaluation/new-run
```

生成匿名的 `inputs/manifest.json`、独立 `labels.jsonl` 和 `splits.json`。固定种子 20261107，按学生分组为 5 名调试、10 名初评、35 名留出。该命令是这份 ZIP 的适配器；通用网站接口不要求这个文件命名或人数。

学生页图转为无元数据 PNG，遮蔽第一张答题卡左上方身份区域；主观题区域不被遮蔽。全部输入、标签及分组在私有暂存目录准备成功后一次发布，出错不留下可被模型读取的半成品清单。原 ZIP 保持不变。

成绩只在 `evaluate` 或 `benchmark` 的模型流程结束后读取，不进入模型输入、细则构建及自动复判路由。初评与留出名单在首次真实评分前已保存。

## 提交与对照

```bash
# 新试卷：50元为该试卷细则准备及全部后续批次的累计总预算；5元为本批次预算
.venv/bin/ai4edu submit data/evaluation/new-run/inputs/manifest.json \
  --server http://127.0.0.1:18000 --exam-budget 50 --budget 5 --strategy C \
  --candidate-ids candidate_0015 candidate_0027 --output data/runs/new-run/debug.json

# 已有细则 ready 后，提交初评名单；名单从 splits.json 的 initial 读取
# --exam-id 指向已经冻结的同一套评分细则；此时沿用已有总预算
.venv/bin/ai4edu submit data/evaluation/new-run/inputs/manifest.json \
  --server http://127.0.0.1:18000 --exam-id EXAM_ID --budget 20 --strategy C \
  --candidate-ids INITIAL_CANDIDATE_IDS --output data/runs/new-run/source.json

# 等全部考生输入准备完成，再提交相同裁图上的 Max 对照；Celery 自动执行
.venv/bin/ai4edu benchmark --source-job SOURCE_C_JOB_ID \
  --labels data/evaluation/new-run/labels.jsonl --split initial \
  --output data/runs/new-run/comparison --budget 20 --wait-seconds 3600
```

上面的 `EXAM_ID`、`SOURCE_C_JOB_ID` 和 `INITIAL_CANDIDATE_IDS` 替换为提交结果与固定分组。`benchmark` 要求源批次是 Plus 主评、Max 自动复判的 risk 配置。它保存 `submission.json`，命令中断或等待超时后，后台已提交任务仍可继续；再次运行同一输入会复用原对照批次。

源批次的独立初评检查点构成 B，源批次最终结果构成实际 C。A 使用相同题目、细则及已准备的同一套图片，由 Max 独立执行转写与判分，不接收 Plus 结论。报告同时保存 C 的离线模拟，实际实验支出与模拟生产费用分开记录。

每个批次保存模型角色、图片预算、输出上限、思考开关、价格、复判参数与提示词版本；同一代码版本的失败单元可重试。提示词版本变化后，旧批次不能复用旧检查点继续生成新版本结果，应创建新批次。

```bash
# 只读重建报告，不新增模型调用
.venv/bin/ai4edu evaluate --source-job SOURCE_C_JOB_ID --max-job MAX_JOB_ID \
  --labels data/evaluation/new-run/labels.jsonl --split initial \
  --output data/runs/new-run/comparison
```

输出包含逐题 A/B/C 及 C_simulated JSONL、`report.json` 和 `report.md`。失败、无法判定及缺失记录保留在覆盖率分母中；所有候选均无法判分时仍输出报告，选择状态为证据不足。正分必须有清楚的学生证据，无法辨认的作答不自动计为零分。

## 本轮已经跑通的真实闭环

- 共享试卷：`5160a46f43914a41b6b934072c6d4b4f`，第15—19题，固定细则满分分别为13、15、15、17、17。
- 修正后调试批次：`39f85ff7bb584285af3555526c31a4dc`，2名学生、10题，10题全部 scored。
- 生效提示词版本 v1.2，所有阶段关闭思考，每次输出上限4096，图片预算2,621,440像素。
- 40次真实调用；按配置参考单价估算费用2.340158元，包括该批次定位和自动复判，不包括此前共享细则及第一轮调试成本。
- 调试 B 与 C 的10题 MAE 均为0.85分、±1分比例均为80%。调试集参与了协议修改，不能当作独立测试集。
- 原始失败调试保留在 `data/runs/functional-v1/debug`；修正后完整输出和离线调试指标在 `data/runs/functional-v2/debug`。

本轮试卷总预算为50元，包含已有调试调用。实际账单以百炼为准；没有 usage 的调用保留费用预留，不当作免费。
