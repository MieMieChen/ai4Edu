# AI4Edu 主观题评分后端

面向手写数学等主观题的自动评分服务。输入题目、参考解答、各题满分，以及按题图片或整份答卷；输出逐评分点得分、学生证据、理由、考生成绩、任务进度和模型费用记录。没有教师审核步骤。

当前实现服务端与命令行工具，网站可通过 `/v1` 接口接入。已完成真实 PostgreSQL/Redis/Celery/API 与百炼闭环，以及同一细则和裁图下10名学生、50道主观题的 A/B/C 对照。当前暂选 B：Plus 独立转写及主评，Max 生成并核验统一细则；A/B/C 可按批次切换。这个结论适用于目前这套试卷，跨试卷仍需验证。模型 ID 在配置中指定，不因错误自动替换。

运行与实验记录见 [本机运行手册](docs/functional-runbook.md)、[验证记录](docs/verification.md) 和 [千人并发与故障测试](docs/concurrency-loadtest.md)。网站前端尚未建设。

## 启动

需要 Python 3.11+、uv；完整服务需要 PostgreSQL、Redis 和支持 Compose 的 Docker 环境。

```bash
cp .env.example .env
# 确认 bailian.txt 中有 api_key、base_url；编辑 .env 中的模型、参考单价与账号限额。
uv sync --extra dev
docker compose up --build -d
```

容器会显式创建数据库表，随后启动 API、输入准备 Worker、评分 Worker、调度 Worker 和 Celery Beat。接口文档：<http://127.0.0.1:8000/docs>。Compose 将凭据文件作为只读 secret 挂载；文件同时被 Git、Docker 构建上下文和 Python 分发包排除。`AI4EDU_SERVICE_API_KEY` 为网站调用这个后端的独立 Bearer 凭据，设置后所有请求均需携带。

也可只用 Docker 启动依赖，并在本机调试：

```bash
docker compose up -d postgres redis
uv run ai4edu init-db
uv run uvicorn ai4edu.api:create_app --factory --reload
# 另开终端运行 Worker 和 Beat：
uv run celery -A ai4edu.tasks:celery worker -Q preparation,grading,control --pool threads --concurrency 8 --loglevel INFO
uv run celery -A ai4edu.tasks:celery beat --schedule data/celerybeat-schedule --loglevel INFO
```

默认开发文件存储为 `data/assets`，多个本机服务共享此目录。容器使用共享卷；跨机器扩容时将 `AI4EDU_STORAGE_BACKEND=oss` 并填写 OSS 配置。数据库不会根据连接失败自动切换 SQLite；SQLite 仅供显式配置的本地测试使用。

## 百炼配置

| 配置 | 含义 |
| --- | --- |
| `AI4EDU_CREDENTIALS_FILE` | 显式指定包含 `api_key=...` 与 `base_url=...` 的文件；已配置的环境凭据优先；解析错误不输出原始行 |
| `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` | 百炼兼容接口凭据与完整 `/v1` 基础地址；也识别 `OPENAI_API_KEY`、`OPENAI_BASE_URL` |
| `AI4EDU_MODEL` | 显式指定的公共模型 ID |
| `AI4EDU_LAYOUT_MODEL` | 方向判断与整卷题目定位模型 |
| `AI4EDU_RUBRIC_MODEL` | 评分细则构建与核验模型 |
| `AI4EDU_TRANSCRIBE_MODEL`、`AI4EDU_GRADE_MODEL` | 独立转写与主评分模型 |
| `AI4EDU_VERIFY_MODEL` | 独立复判模型；角色未单设时复用公共模型 |
| `AI4EDU_MODEL_PRICES` | 模型 ID 到输入/输出单价的 JSON 映射，单位为元/百万 Token；示例 `{"chosen-model":{"input":2,"output":8}}`，示例数值不是当前模型报价 |
| `AI4EDU_MODEL_CONCURRENCY`、`AI4EDU_MODEL_RPM`、`AI4EDU_MODEL_TPM` | Redis 统一控制的模型在途请求、分钟请求数和 Token 额度 |
| `AI4EDU_MAX_OUTPUT_TOKENS`、`AI4EDU_MAX_PIXELS` | 单次输出上限与每张处理图片的像素预算 |
| `AI4EDU_MAX_TRANSCRIPTION_TOKENS`、`AI4EDU_MAX_GRADING_TOKENS` | 转写与评分分别设置输出上限；当前为8192/4096，包含Max独立复判的相应阶段 |
| `AI4EDU_MAX_INPUT_TOKENS` | 请求前保守估算的输入预算；当前250000，超限时不发请求、不截断后继续评分 |
| `AI4EDU_TRANSCRIPTION_IMAGES_PER_CALL`、`AI4EDU_TRANSCRIPTION_CONCURRENCY` | 长题按自然图片分片，每题最多2个并行转写请求；统一合并证据后对整题评分 |
| `AI4EDU_REGION_PADDING_RATIO` | 整页定位框按短边像素留4%边距，受邻题间隔约束；保留原始检测框及实际裁图框 |
| `AI4EDU_VERIFICATION_POLICY` | `off`：A/B 两阶段对照；`all`：全部独立复判；`risk`：歧义/未解决项/问题标记触发复判，另加确定性抽样 |
| `AI4EDU_AUDIT_RATE` | risk 模式中普通作答的抽样比例，默认 0.1；属于待评测策略，不代表已验证的质量阈值 |
| `AI4EDU_THINKING_ROLES`、`AI4EDU_THINKING_BUDGET` | 显式开启思考的角色列表及预算；默认关闭。参数改变会改变成功缓存指纹，真实模型测试应记录生效配置 |

调用要求模型支持图像输入和 JSON Object 输出。协议或思考参数不兼容会明确报错。默认方向判断与版面定位合并为一次视觉调用；裁图、旋转与像素预算由 OpenCV/Pillow 执行。`images.PaddleOrientation` 提供独立的本地方向分类组件，安装 `uv sync --extra paddle` 后可在集成中显式使用；当前默认流程不下载 Paddle 模型。

进行 A/B 对照时，A 将转写与判分角色设为 Max，B 设为 Plus，均使用 `verification_policy=off`。C 使用 Plus 主评、Max 复判和 `risk`。三种策略需要复用已经冻结的同一细则及同一套裁图，不能将分别生成的细则或裁图混为公平对照。`scored` 对应评测方案中的 `resolved` 含义。

## 数据与流程

```text
试卷：题目/参考图片与文字 + 已知满分
  → 评分点草稿 → 独立校核 → 程序验证分值 → 固定评分细则

学生输入：按题图片 或 按页排序的整卷图片/PDF
  → 方向与区域识别 → 原图坐标裁切、转正、分辨率控制
  → 不接触参考答案的忠实转写
  → 原图 + 转写 + 题目 + 固定细则主评
  → 必要时重新转写并独立复判（不提供初评分数）
  → 分值/评分点覆盖/证据引用校验 → 保存结果
```

同一试卷的评分细则准备一次，后续批次复用。程序按 `Decimal` 累加分数；模型不负责提供总分。正向给分必须引用存在且清楚的学生证据；划掉的内容和无法辨认的符号不能直接作为确定给分的依据。空白作答与无法辨认分别处理。信息不足时对应分项和总分为 `null`，状态为 `ungradable`。

每个批次固定模型、提示词版本、输出预算、分片参数和图像参数。不同学生及题目并行执行，同题保留转写后评分的依赖。默认32个评分执行线程、4个输入准备线程，每模型统一控制最多16个在途请求；多个Worker共用Redis准入和速率限制。长题成功分片会保存检查点，一片出错时先等待已经发出的请求结束并保存成功结果，再记录失败或容量等待。

评分细则生成失败、模型 JSON 错误、题号不匹配、证据错误、得分超过上限等直接记录为 `failed`。没有 JSON 修复、默认零分、运行时模拟评分或静默换模型。模型请求本身不自动重试；可以通过批次重试接口重新执行失败单元。Redis 容量不足时延迟调度，属于请求发出前的排队。

## 接口顺序

1. `POST /v1/assets`：multipart 字段 `file`，接受 JPEG/PNG/WebP/PDF；返回有序图片 ID 列表。PDF 按 150 DPI 展页。
2. `POST /v1/exams`：提交主观题题目、参考解答和满分，后台自动准备评分细则。
3. `GET /v1/exams/{id}`：查看 `queued/running/ready/failed`、最终细则和错误。
4. `POST /v1/jobs`：试卷 ready 后提交最多1000名考生；立即返回批次 ID 和进度。
5. `GET /v1/jobs/{id}`：查看完成数、状态、用量、预算与费用。
6. `GET /v1/jobs/{id}/results?offset=0&limit=50`：按考生分页查看成绩、逐题结果及原图区域映射。
7. `GET /v1/jobs/{id}/export`：流式 JSONL 导出，保留未完成或失败状态。
8. `POST /v1/jobs/{id}/retry`、`POST /v1/exams/{id}/retry`：重试失败单元，保留通过校验的已有阶段结果。

试卷请求示例：

```json
{
  "title": "数学主观题",
  "score_step": "0.5",
  "questions": [
    {
      "id": "15",
      "stem": "完整题干，可用 question_assets 中的图片代替",
      "question_assets": [],
      "reference": "完整参考解答，可用 reference_assets 中的图片代替",
      "reference_assets": [],
      "max_score": "13"
    }
  ]
}
```

整卷请求示例（`pages` 必须属于同一考生并按阅读顺序排列）：

```json
{
  "exam_id": "exam-id",
  "budget_cny": "100",
  "candidates": [
    {"candidate_id": "c001", "kind": "paper", "pages": ["page-1-id", "page-2-id"]}
  ]
}
```

按题输入将该考生替换为下面的结构，每道主观题允许多张作答图：

```json
{
  "candidate_id": "c001",
  "kind": "question",
  "answers": [{"question_id": "15", "assets": ["answer-image-id"]}]
}
```

按题输入需要覆盖本试卷的所有主观题。需要只判部分题时，为这些题创建对应试卷配置。提交批次支持 `Idempotency-Key` 请求头；相同键与相同请求内容返回同一批次，内容冲突返回409。

每题结果包含：`grade`（逐评分点得分、理由和证据 ID）、`transcript`（证据原文与来源图片）、`verified`、`primary_grade`、`rubric_version`、`prompt_version`。`source_regions` 记录 `source_asset_id`、原图归一化 `bbox`、顺时针校正角度、页序与区域序。通过 `/v1/assets/{id}` 查看原图或裁图。精确分值在评分结构中序列化为十进制字符串。

## 1000名考生与费用

每人5道题的一批输入会持久化为1000个输入准备任务和5000个评分单元，两个业务队列可分别扩容。HTTP 请求不等待模型完成。数据库 Outbox 与批次在同一个事务提交，消息发送成功后才删除。容量等待同样通过有执行时间的 Outbox 持久化，不占用 Celery 的失败重试次数。执行令牌保护检查点与最终结果，旧 Worker 不能覆盖新执行。调度队列独立，避免长时间模型调用阻塞分发。

评分请求成功通过结构及业务校验后可以缓存复用。普通输入处理失败会保留已成功的页级检查点；定位缺题的显式重试才重新执行布局识别。Worker 异常中断后按租约恢复；超时请求仍可能在供应商侧执行，不能保证供应商恰好计费一次。HTTP 超时和服务失败不会自动循环调用。真实处理时间需要结合所选模型、账号限额和输出量压测。

所有调用保存模型、角色、原始响应、usage、状态与费用。试卷准备的调用以 exam_id 关联；学生处理阶段以 job_id 关联。模型价格未配置时费用保持未知，并在 `unpriced_or_unknown_calls` 中反映，不能视作免费。

`POST /v1/exams` 可设置 `budget_cny`，它覆盖共享细则及该试卷全部批次的累计费用；批次预算是其下的进一步限制。并发调用同时预留试卷与批次预算，取得usage后结算；中断且用量未知的调用保留预留和审计。

## 已完成的实测与数据

- 真实调试闭环：2名学生、10题全部评分；另有4名留出学生20题，其中17题评分、3题无法判定，没有执行失败。
- 模型初评：A 43/50可判定，B 46/50，C 44/50。在B/C共同可判定的43题上，MAE分别约1.198与1.140，C判题流程费用约为B的4.48倍，当前暂选B。
- 长题回归：原来4096输出Token截断的同一张图，在8192上限下完整输出4871Token并完成评分；两张作答图的并行转写也完成，结果及调用记录保留。
- 真实模型调用实测峰值16；1000人模拟模型全链路测试5000题全部正确、唯一、可查询，实际模拟HTTP峰值18；Worker强制终止后自然租约恢复约301秒。
- 当前代码测试270项通过。模拟模型吞吐用于容量验证，不能推算百炼真实处理速度；现有CSV只有题目总分，不能证明逐步骤数学判断正确。

仓库包含源码、测试、依赖配置和技术文档。原始 ZIP、学生试卷与成绩、整个 `data/` 目录、数据库快照和逐次模型输出均只保留在本地，不随代码上传；运行工具会在本机生成所需目录。下文使用样例的命令需要自行提供本地数据文件。

本机的 `data/evaluation/2026-11-07` 保存匿名输入、独立标签和固定5/10/35学生分组；`data/runs` 保存真实输出及模型比较；`data/loadtest/20261107_concurrency32b` 保存千人及故障报告。API密钥、`.env`、`bailian.txt`、虚拟环境与运行状态同样排除。

配置预算需要同时配置所用模型单价。请求前按文本、图像和输出预算预留费用；拿到 usage 后按配置单价结算。没有 usage 的失败请求保留预留额，避免把未知消费重新花掉。该值是预算控制估算，缓存折扣、实际视觉分词与供应商计费以百炼账单为准。批次的费用统计不包含已复用的试卷准备成本；固定成本可查询数据库 `model_calls` 的 exam_id。

## 使用现有样例

```bash
uv run ai4edu prepare-sample --archive 测试数据集.zip --output data/sample --limit 2
# 服务和 Worker 启动、模型配置完成后：
uv run ai4edu submit data/sample/manifest.json --server http://127.0.0.1:8000
```

`prepare-sample` 是这份 ZIP 的适配器：导出第15—19题题目与参考解答、匿名考生页图和 manifest。不会把文件名中的总分、姓名、学号或成绩 CSV 写入模型输入。原始 ZIP 和原有模型评测方案保持不变。首次建议仅2名考生检查真实调用及费用；`--limit 50` 可整理全部样例。

## 测试与验证边界

可单独探测两个候选模型，不上传学生图像：

```bash
uv run ai4edu probe --output data/probe-v1.1.json
```

该命令每轮最多两个模型，每次使用合成等式图片、关闭思考、最多128输出 Token，输出权限/格式/用量信息；失败响应不打印凭据、请求地址或原始错误体。这不是判分准确率评测。

```bash
uv run pytest
uv run ruff check src tests
uv run python -m compileall -q src
```

测试覆盖实际图片旋转/透明通道/裁切/PDF、评分点和证据约束、1000名考生拆出5000条评分单元、共享Redis协议限流、预算拒绝、Outbox发送失败、租约恢复、断点复用及完整服务流程。数据库集成测试显式使用启用外键约束的 SQLite；Redis 协议测试使用 fakeredis。外部模型只在测试中由固定服务替身代替，测试通过不代表真实手写判分准确率或百炼生产吞吐已达标。

上线前仍应使用选型 session 确定的模型与价格，在真实 PostgreSQL、Redis 和百炼环境运行跨试卷盲测及1000人压测，记录误差、无法判定率、总耗时和账单费用。
