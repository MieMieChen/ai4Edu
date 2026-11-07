"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const PAGE_SIZE = 50;
  const TERMINAL_JOBS = new Set(["completed", "completed_with_errors", "failed", "cancelled"]);
  const STATE_LABELS = {
    ready: "已就绪", waiting: "等待准备", queued: "排队中", running: "处理中", preparing: "准备中",
    scored: "已评分", resolved: "已评分", completed: "已完成",
    completed_with_errors: "完成 · 含失败", ungradable: "无法评分", failed: "失败",
    budget_exceeded: "预算不足", missing: "结果缺失", cancelled: "已取消",
  };
  let storedToken = "";
  try { storedToken = sessionStorage.getItem("ai4edu-service-token") || ""; } catch (_) { /* Optional tab storage. */ }

  const state = {
    token: storedToken, jobs: [], exams: [], jobId: null, summary: null, exam: null,
    candidates: [], candidateId: null, questionId: null, offset: 0, evaluation: null,
    busy: false, metaSequence: 0, viewSequence: 0, metaController: null,
    viewController: null, pollTimer: null, pollRequest: null, pollPausedError: false,
    sources: [], sourceKey: null, sourceId: null, imageSequence: 0, imageController: null,
    imageUrls: new Map(), activeImageUrl: null, zoom: 1, highlightedLine: null,
    materialKey: null, materialSequence: 0, materialController: null, materialUrls: new Set(),
    submitting: false, submissionUncertain: false,
  };

  class APIError extends Error {
    constructor(message, status) { super(message); this.name = "APIError"; this.status = status; }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function icon(name) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    node.setAttribute("class", "icon");
    node.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    node.append(use);
    return node;
  }

  function numeric(value) {
    if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function number(value, maximumFractionDigits = 3) {
    const parsed = numeric(value);
    return parsed === null ? "—" : new Intl.NumberFormat("zh-CN", { maximumFractionDigits }).format(parsed);
  }

  function money(value) {
    const parsed = numeric(value);
    return parsed === null ? "—" : `¥${new Intl.NumberFormat("zh-CN", {
      minimumFractionDigits: 2, maximumFractionDigits: 8,
    }).format(parsed)}`;
  }

  function percent(value) {
    const parsed = numeric(value);
    return parsed === null ? "—" : new Intl.NumberFormat("zh-CN", {
      style: "percent", maximumFractionDigits: 1,
    }).format(parsed);
  }

  function text(value, fallback = "未提供") {
    if (value === null || value === undefined || value === "") return fallback;
    if (typeof value === "string") return value;
    if (typeof value === "object") {
      try { return JSON.stringify(value); } catch (_) { return fallback; }
    }
    return String(value);
  }

  function dateLabel(value) {
    if (!value) return "时间未记录";
    // Database timestamps without a zone are stored in UTC.
    const source = String(value);
    const normalized = /^\d{4}-\d\d-\d\d[T ]\d\d:\d\d/.test(source) && !/(Z|[+-]\d\d:\d\d)$/i.test(source)
      ? `${source.replace(" ", "T")}Z` : source;
    const date = new Date(normalized);
    if (Number.isNaN(date.valueOf())) return "时间未记录";
    return new Intl.DateTimeFormat("zh-CN", {
      month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
    }).format(date);
  }

  function stateLabel(value) { return STATE_LABELS[value] || text(value, "状态未知"); }
  function tagClass(value) {
    if (["scored", "resolved", "completed", "ready"].includes(value)) return "success";
    if (["failed", "completed_with_errors"].includes(value)) return "error";
    if (["ungradable", "budget_exceeded", "missing"].includes(value)) return "warning";
    return "running";
  }

  function setTag(node, value) { node.textContent = stateLabel(value); node.className = `tag ${tagClass(value)}`; }
  function aborted(error) { return error && error.name === "AbortError"; }

  function detailMessage(detail) {
    if (Array.isArray(detail)) return detail.map((item) => {
      if (!item || typeof item !== "object") return text(item, "请求参数无效");
      const location = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
      return `${location ? `${location}：` : ""}${text(item.msg, "请求参数无效")}`;
    }).join("；");
    return text(detail, "服务未返回错误说明");
  }

  async function request(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
    if (!headers.has("Accept")) headers.set("Accept", "application/json");
    const { responseType = "json", ...fetchOptions } = options;
    let response;
    try {
      response = await fetch(path, { ...fetchOptions, headers, cache: "no-store", credentials: "same-origin" });
    } catch (error) {
      if (aborted(error)) throw error;
      throw new APIError("无法连接本地服务，请检查服务是否正在运行。", null);
    }
    if (!response.ok) {
      let message = `请求失败（HTTP ${response.status}）`;
      try {
        const body = await response.json();
        if (body && body.detail !== undefined) message = detailMessage(body.detail);
      } catch (_) { /* Keep the status when the server returns a non-JSON error. */ }
      throw new APIError(message, response.status);
    }
    if (responseType === "blob") return response.blob();
    try { return await response.json(); }
    catch (error) {
      if (aborted(error)) throw error;
      throw new APIError("服务返回了无法读取的数据，请重新读取运行记录。", response.status);
    }
  }

  function connection(kind, message) {
    $("connectionState").className = `connection-state${kind ? ` is-${kind}` : ""}`;
    $("connectionText").textContent = message;
  }

  function showAlert(message, error) {
    $("alertMessage").textContent = message;
    $("appAlert").hidden = false;
    $("alertAuthButton").hidden = !(error && error.status === 401);
    if (error && error.status === 401) {
      connection("error", "需要服务凭据");
      $("authDetails").open = true;
    }
  }

  function clearAlert() { $("appAlert").hidden = true; }

  function focusAuth() {
    $("authDetails").open = true;
    $("serviceToken").scrollIntoView({ block: "center", behavior: "smooth" });
    $("serviceToken").focus({ preventScroll: true });
  }

  function setBusy(value) {
    state.busy = value;
    $("refreshButton").disabled = value;
    $("refreshButton").setAttribute("aria-busy", String(value));
    $("newJobButton").disabled = value || state.submitting;
    $("candidateSelect").disabled = value || !state.candidates.length;
    $("questionSelect").disabled = value || !currentQuestionIds().length;
    renderPagination();
  }

  function showLoading() {
    $("loadingState").hidden = false;
    $("workspace").hidden = true;
    $("emptyState").hidden = true;
  }

  function showEmpty(title, description) {
    $("loadingState").hidden = true;
    $("workspace").hidden = true;
    $("emptyState").hidden = false;
    $("emptyTitle").textContent = title;
    $("emptyDescription").textContent = description;
  }

  function currentJob() { return state.jobs.find((job) => job.id === state.jobId) || null; }
  function currentCandidate() { return state.candidates.find((candidate) => candidate.candidate_id === state.candidateId) || null; }
  function currentAnswers() {
    const candidate = currentCandidate();
    return candidate && Array.isArray(candidate.answers) ? candidate.answers : [];
  }
  function examQuestions(exam = state.exam) { return exam && Array.isArray(exam.questions) ? exam.questions : []; }
  function currentQuestionIds() {
    const answers = currentAnswers();
    return [...new Set((answers.length ? answers : examQuestions()).map((item) => String(item.question_id ?? item.id)))];
  }
  function currentAnswer() { return currentAnswers().find((answer) => String(answer.question_id) === state.questionId) || null; }
  function currentQuestion() { return examQuestions().find((question) => String(question.id) === state.questionId) || null; }
  function currentRubric() {
    const rubrics = state.exam && state.exam.rubrics;
    if (Array.isArray(rubrics)) return rubrics.find((rubric) => String(rubric.question_id) === state.questionId) || null;
    if (rubrics && typeof rubrics === "object" && Object.hasOwn(rubrics, state.questionId)) return rubrics[state.questionId];
    return null;
  }
  function questionName(identifier) { return /^\d+$/.test(String(identifier)) ? `第 ${identifier} 题` : `题目 ${identifier}`; }

  function renderJobs() {
    $("jobCount").textContent = String(state.jobs.length);
    const nodes = state.jobs.map((job) => {
      const node = el("button", `job-button${job.id === state.jobId ? " is-selected" : ""}`);
      node.type = "button";
      node.title = `${text(job.exam_title, "未命名试卷")} · ${text(job.id)}`;
      node.setAttribute("aria-current", job.id === state.jobId ? "true" : "false");
      node.append(el("span", "job-title", text(job.exam_title, "未命名试卷")));
      const row = el("span", "job-row");
      row.append(el("span", "job-date", dateLabel(job.created_at)));
      const status = el("span", `job-state state-${tagClass(job.state)}`);
      status.append(el("span", "small-dot"), document.createTextNode(stateLabel(job.state)));
      row.append(status);
      node.append(row);
      node.append(el("span", "job-subtitle", `${number(job.candidate_count, 0)} 位考生 · ${number(job.answer_count, 0)} 份作答`));
      node.addEventListener("click", () => {
        if (job.id === state.jobId && state.summary && !state.pollPausedError) return;
        loadJob(job.id, { offset: 0 });
      });
      return node;
    });
    $("jobList").replaceChildren(...(nodes.length ? nodes : [el("p", "sidebar-empty", "尚无运行记录")]));
  }

  function renderHeading() {
    const job = currentJob();
    $("examTitle").textContent = text(state.exam && state.exam.title, text(job && job.exam_title, "主观题评分工作台"));
    $("batchDescription").textContent = job
      ? `${dateLabel(job.created_at)} · ${number(job.candidate_count, 0)} 位考生 · ${number(job.answer_count, 0)} 份作答`
      : "已保存的评分记录 · 查看作答图片、评分标准与证据";
  }

  function renderMetrics() {
    const summary = state.summary || {};
    const counts = summary.counts && typeof summary.counts === "object" ? summary.counts : null;
    $("progressValue").textContent = `${number(summary.completed, 0)} / ${number(summary.total, 0)}`;
    setTag($("batchState"), summary.state);
    $("progressDescription").textContent = counts
      ? `${number(counts.scored ?? 0, 0)} 已评分 · ${number(counts.ungradable ?? 0, 0)} 无法评分 · ${number(counts.failed ?? 0, 0)} 失败`
      : "处理状态未记录";
    $("costValue").textContent = money(summary.known_cost_cny);
    const unknown = numeric(summary.unpriced_or_unknown_calls);
    const budget = numeric(summary.budget_cny);
    const costNote = unknown === null ? "费用完整性未知" : unknown > 0 ? `${number(unknown, 0)} 次调用费用未知` : "全部调用已计价";
    $("costDescription").textContent = `${costNote}${budget !== null ? ` · 预算 ${money(budget)}` : ""}`;
    $("callsValue").textContent = number(summary.model_calls, 0);
    const unknownUsage = numeric(summary.unknown_usage_calls);
    $("tokensDescription").textContent = `已记录输入 ${number(summary.input_tokens, 0)} / 输出 ${number(summary.output_tokens, 0)} tokens${unknownUsage > 0 ? ` · ${number(unknownUsage, 0)} 次用量未知` : ""}`;
    const evaluation = state.evaluation;
    if (evaluation && evaluation.available) {
      const metrics = evaluation.metrics || {};
      $("qualityValue").textContent = percent(evaluation.all_expected_within_1_rate);
      $("qualityDescription").textContent = `全部预期 ${number(metrics.expected_count, 0)} 题 · 小样本离线参考`;
    } else {
      $("qualityValue").textContent = "—";
      $("qualityDescription").textContent = "暂无可用离线评估 · 查看说明";
    }
    const config = summary.execution_config || {};
    const job = currentJob() || {};
    const model = config.grade_model || job.grade_model || config.model;
    let thinking = job.thinking;
    if (Array.isArray(config.thinking_roles)) thinking = config.thinking_roles.includes("grade");
    if (thinking && typeof thinking === "object" && typeof thinking.grade === "boolean") thinking = thinking.grade;
    const thinkingText = thinking === true ? "评分思考开启" : thinking === false ? "评分思考关闭" : null;
    const verification = config.verification_policy === "off" ? "未启用复核" : config.verification_policy === "risk" ? "按风险复核" : null;
    $("runtimeMeta").textContent = [model ? `评分模型 ${text(model)}` : "评分模型未记录", thinkingText, verification].filter(Boolean).join(" · ");
    $("jobIdentifier").textContent = `批次 ${text(state.jobId, "未知")}`;
  }

  function renderPagination() {
    const job = currentJob();
    const total = numeric(job && job.candidate_count);
    const length = state.candidates.length;
    const range = length ? `${number(state.offset + 1, 0)}–${number(state.offset + length, 0)}` : "0";
    $("candidatePage").textContent = `${range} / ${number(total, 0)} 位`;
    $("previousPage").disabled = state.busy || state.offset === 0;
    $("nextPage").disabled = state.busy || (total !== null ? state.offset + length >= total : length < PAGE_SIZE);
  }

  function renderSelectors() {
    if (!state.candidates.some((candidate) => candidate.candidate_id === state.candidateId)) {
      state.candidateId = state.candidates.length ? state.candidates[0].candidate_id : null;
    }
    const candidateOptions = state.candidates.map((candidate) => {
      const score = numeric(candidate.total_score);
      const option = el("option", "", `${text(candidate.candidate_id)}${score === null ? " · 总分未完成" : ` · 总分 ${number(score)}`}`);
      option.value = candidate.candidate_id;
      return option;
    });
    if (!candidateOptions.length) candidateOptions.push(el("option", "", "本页暂无考生"));
    $("candidateSelect").replaceChildren(...candidateOptions);
    if (state.candidateId !== null) $("candidateSelect").value = state.candidateId;
    const identifiers = currentQuestionIds();
    if (!identifiers.includes(state.questionId)) state.questionId = identifiers[0] || null;
    const questionOptions = identifiers.map((id) => {
      const question = examQuestions().find((item) => String(item.id) === id);
      const maximum = question && question.max_score;
      const option = el("option", "", `${questionName(id)}${numeric(maximum) !== null ? ` · ${number(maximum)} 分` : ""}`);
      option.value = id;
      return option;
    });
    if (!questionOptions.length) questionOptions.push(el("option", "", "暂无题目"));
    $("questionSelect").replaceChildren(...questionOptions);
    if (state.questionId !== null) $("questionSelect").value = state.questionId;
    $("candidateSelect").disabled = state.busy || !state.candidates.length;
    $("questionSelect").disabled = state.busy || !identifiers.length;
    renderPagination();
  }

  function renderQuestion() {
    const question = currentQuestion();
    const questionAssets = materialAssetIds(question && question.question_assets);
    const referenceAssets = materialAssetIds(question && question.reference_assets);
    const stem = question && typeof question.stem === "string" ? question.stem : "";
    const reference = question && typeof question.reference === "string" ? question.reference : "";
    $("questionNumber").textContent = state.questionId ? questionName(state.questionId) : "题目";
    $("questionPreview").textContent = stem.trim() ? stem : questionAssets.length
      ? `图片题干 · ${questionAssets.length} 张` : "当前记录未保存题干内容";
    $("questionContent").hidden = !stem.trim() && questionAssets.length > 0;
    $("questionContent").textContent = stem.trim() ? stem : questionAssets.length ? "" : "当前记录未保存题干内容。";
    $("referenceContent").hidden = !reference.trim() && referenceAssets.length > 0;
    $("referenceContent").textContent = reference.trim() ? reference : referenceAssets.length ? "" : "当前记录未保存参考解答内容。";
    renderMaterialImages(questionAssets, referenceAssets);
  }

  function materialAssetIds(value) {
    return Array.isArray(value) ? [...new Set(value.filter((id) => typeof id === "string" && id))] : [];
  }

  function clearMaterialImages() {
    state.materialSequence += 1;
    if (state.materialController) state.materialController.abort();
    state.materialController = null;
    for (const url of state.materialUrls) URL.revokeObjectURL(url);
    state.materialUrls.clear();
    state.materialKey = null;
    for (const container of [$("questionImages"), $("referenceImages")]) {
      container.replaceChildren();
      container.hidden = true;
    }
  }

  function renderMaterialImages(questionAssets, referenceAssets) {
    const key = JSON.stringify([state.exam && state.exam.id, state.questionId, questionAssets, referenceAssets]);
    if (key === state.materialKey) return;
    clearMaterialImages();
    state.materialKey = key;
    const sequence = state.materialSequence;
    const controller = new AbortController();
    state.materialController = controller;
    for (const [container, identifiers, label] of [
      [$("questionImages"), questionAssets, "题干图片"],
      [$("referenceImages"), referenceAssets, "参考解答图片"],
    ]) {
      container.hidden = !identifiers.length;
      identifiers.forEach((identifier, index) => {
        const figure = el("figure", "material-image-card");
        const image = el("img", "material-image");
        image.alt = `${questionName(state.questionId)} · ${label} ${index + 1}`;
        image.hidden = true;
        const caption = el("figcaption", "material-image-caption");
        const status = el("span", "material-image-status", `正在读取${label} ${index + 1}…`);
        status.setAttribute("role", "status");
        const openButton = el("button", "text-button", "查看原尺寸");
        openButton.type = "button";
        openButton.disabled = true;
        openButton.append(icon("chevron"));
        caption.append(status, openButton);
        figure.append(image, caption);
        container.append(figure);
        loadMaterialImage(identifier, image, status, openButton, `${label} ${index + 1}`, sequence, controller.signal);
      });
    }
  }

  async function loadMaterialImage(identifier, image, status, openButton, label, sequence, signal) {
    try {
      const blob = await request(`/v1/assets/${encodeURIComponent(identifier)}`, { signal, responseType: "blob" });
      if (sequence !== state.materialSequence || signal.aborted) return;
      const url = URL.createObjectURL(blob);
      state.materialUrls.add(url);
      image.onload = () => {
        if (sequence !== state.materialSequence || signal.aborted) return;
        image.hidden = false;
        status.textContent = label;
        openButton.disabled = false;
      };
      image.onerror = () => {
        if (sequence !== state.materialSequence || signal.aborted) return;
        image.hidden = true;
        status.textContent = `${label}无法解码，请重新读取运行记录。`;
        status.classList.add("is-error");
      };
      openButton.addEventListener("click", () => {
        if (sequence === state.materialSequence && !signal.aborted) window.open(url, "_blank", "noopener,noreferrer");
      });
      image.src = url;
    } catch (error) {
      if (aborted(error) || sequence !== state.materialSequence) return;
      status.textContent = `${label}读取失败：${error.message}`;
      status.classList.add("is-error");
      if (error.status === 401) showAlert("读取题干与参考解答图片需要有效的服务访问凭据。", error);
    }
  }

  function transcriptLines() {
    const answer = currentAnswer();
    const transcript = answer && answer.result && answer.result.transcript;
    return transcript && Array.isArray(transcript.lines) ? transcript.lines : [];
  }

  function sourcesFor(answer) {
    const sources = [];
    const seen = new Set();
    if (answer && Array.isArray(answer.source_regions)) {
      for (const region of answer.source_regions) {
        const source = typeof region === "string" ? { asset_id: region } : region;
        if (!source || !source.asset_id || seen.has(source.asset_id)) continue;
        seen.add(source.asset_id);
        sources.push(source);
      }
    }
    for (const line of transcriptLines()) {
      if (!line.image_id || seen.has(line.image_id)) continue;
      seen.add(line.image_id);
      sources.push({ asset_id: line.image_id });
    }
    return sources;
  }

  function sourceLabel(source, index) {
    const page = numeric(source.page_index);
    return page !== null ? `第 ${page + 1} 页${state.sources.filter((item) => item.page_index === source.page_index).length > 1 ? ` · ${index + 1}` : ""}` : `作答图 ${index + 1}`;
  }

  function renderSource() {
    state.sources = sourcesFor(currentAnswer());
    const key = `${state.candidateId}\u0000${state.questionId}`;
    if (key !== state.sourceKey) {
      state.sourceKey = key;
      state.sourceId = null;
      state.zoom = 1;
      state.highlightedLine = null;
    }
    const selected = state.sources.find((source) => source.asset_id === state.sourceId) || state.sources[0];
    const tabs = state.sources.map((source, index) => {
      const node = el("button", `source-tab${selected && source.asset_id === selected.asset_id ? " is-selected" : ""}`, sourceLabel(source, index));
      node.type = "button";
      node.title = `图片 ${text(source.asset_id)}`;
      node.setAttribute("aria-pressed", String(Boolean(selected && source.asset_id === selected.asset_id)));
      node.addEventListener("click", () => showImage(source.asset_id));
      return node;
    });
    $("sourceTabs").replaceChildren(...tabs);
    if (selected) showImage(selected.asset_id);
    else {
      cancelImage();
      state.sourceId = null;
      state.activeImageUrl = null;
      $("answerImage").hidden = true;
      $("imageFeedback").hidden = false;
      $("imageFeedback").textContent = "当前作答暂无可用图片";
      $("sourceInfo").textContent = "未保存图片来源";
      $("openImageButton").disabled = true;
      updateZoomButtons();
    }
  }

  function cancelImage() {
    state.imageSequence += 1;
    if (state.imageController) state.imageController.abort();
    state.imageController = null;
  }

  function clearImageCache() {
    cancelImage();
    clearMaterialImages();
    for (const url of state.imageUrls.values()) URL.revokeObjectURL(url);
    state.imageUrls.clear();
    state.activeImageUrl = null;
    $("answerImage").removeAttribute("src");
  }

  async function showImage(identifier) {
    const source = state.sources.find((item) => item.asset_id === identifier);
    if (!source) return;
    const sameImage = state.sourceId === identifier && state.activeImageUrl && !$("answerImage").hidden;
    state.sourceId = identifier;
    for (const [index, node] of [...$("sourceTabs").children].entries()) {
      const selected = state.sources[index] && state.sources[index].asset_id === identifier;
      node.classList.toggle("is-selected", Boolean(selected));
      node.setAttribute("aria-pressed", String(Boolean(selected)));
    }
    const index = state.sources.indexOf(source);
    const provenance = [sourceLabel(source, index)];
    if (source.bbox) provenance.push("按题目区域裁切");
    else if (source.source_asset_id && source.source_asset_id !== source.asset_id) provenance.push("按题图片规范化");
    if (numeric(source.rotation) !== null && Number(source.rotation) !== 0) provenance.push(`已转正 ${number(source.rotation)}°`);
    $("sourceInfo").textContent = provenance.join(" · ");
    if (sameImage) return;
    cancelImage();
    const sequence = state.imageSequence;
    const controller = new AbortController();
    state.imageController = controller;
    state.zoom = 1;
    state.activeImageUrl = null;
    $("answerImage").hidden = true;
    $("imageFeedback").hidden = false;
    $("imageFeedback").textContent = "正在读取作答图片…";
    $("openImageButton").disabled = true;
    updateZoomButtons();
    try {
      let url = state.imageUrls.get(identifier);
      if (!url) {
        const blob = await request(`/v1/assets/${encodeURIComponent(identifier)}`, { signal: controller.signal, responseType: "blob" });
        if (sequence !== state.imageSequence) return;
        url = URL.createObjectURL(blob);
        // Keep a small tab-local cache; never put service credentials in image URLs.
        while (state.imageUrls.size >= 12) {
          const oldest = state.imageUrls.keys().next().value;
          URL.revokeObjectURL(state.imageUrls.get(oldest));
          state.imageUrls.delete(oldest);
        }
        state.imageUrls.set(identifier, url);
      }
      if (sequence !== state.imageSequence) return;
      const image = $("answerImage");
      image.onload = () => {
        if (sequence !== state.imageSequence) return;
        state.activeImageUrl = url;
        image.hidden = false;
        $("imageFeedback").hidden = true;
        $("openImageButton").disabled = false;
        image.alt = `考生 ${text(state.candidateId)} · ${questionName(state.questionId)} · ${sourceLabel(source, index)}`;
        $("imageViewport").scrollTop = 0;
        $("imageViewport").scrollLeft = 0;
        updateImageSize();
      };
      image.onerror = () => {
        if (sequence !== state.imageSequence) return;
        image.hidden = true;
        $("imageFeedback").hidden = false;
        $("imageFeedback").textContent = "作答图片解码失败，请重新读取运行记录。";
      };
      image.src = url;
    } catch (error) {
      if (aborted(error) || sequence !== state.imageSequence) return;
      $("imageFeedback").textContent = `作答图片读取失败：${error.message}`;
      if (error.status === 401) showAlert("读取作答图片需要有效的服务访问凭据。", error);
    }
  }

  function updateZoomButtons() {
    const ready = Boolean(state.activeImageUrl);
    $("zoomOut").disabled = !ready || state.zoom <= 1;
    $("zoomIn").disabled = !ready || state.zoom >= 3;
    $("zoomReset").disabled = !ready;
    $("zoomReset").textContent = state.zoom === 1 ? "适应" : `${Math.round(state.zoom * 100)}%`;
  }

  function updateImageSize() {
    const image = $("answerImage");
    const viewport = $("imageViewport");
    if (image.hidden || !image.naturalWidth || !viewport.clientWidth) return;
    const padding = window.innerWidth <= 760 ? 36 : 48;
    const width = Math.max(40, viewport.clientWidth - padding);
    const height = Math.max(40, viewport.clientHeight - padding);
    const base = Math.min(width, height * image.naturalWidth / image.naturalHeight, image.naturalWidth);
    image.style.width = `${Math.round(base * state.zoom)}px`;
    updateZoomButtons();
  }

  function renderTranscript() {
    const lines = transcriptLines();
    $("transcriptCount").textContent = `${lines.length} 行`;
    const nodes = lines.map((line, index) => {
      const row = el("div", `transcript-row${line.id === state.highlightedLine ? " is-highlighted" : ""}`);
      row.id = `evidence-line-${index}`;
      row.tabIndex = -1;
      row.dataset.lineId = String(line.id);
      row.append(el("span", "line-id", text(line.id, `行 ${index + 1}`)));
      const content = el("div", "line-content");
      content.append(el("span", `line-text${line.status === "crossed_out" ? " crossed-out" : ""}`, text(line.text, "（空行）")));
      const meta = el("div", "line-meta");
      if (line.status === "unclear") meta.append(el("span", "line-status", "字迹模糊"));
      else if (line.status === "crossed_out") meta.append(el("span", "line-status", "已涂改"));
      else if (line.status && line.status !== "clear") meta.append(el("span", "line-status", text(line.status)));
      if (Array.isArray(line.issues) && line.issues.length) meta.append(el("span", "", line.issues.map((issue) => text(issue)).join("；")));
      if (Array.isArray(line.alternatives) && line.alternatives.length) meta.append(el("span", "", `备选转写：${line.alternatives.map((item) => text(item)).join(" / ")}`));
      if (meta.childNodes.length) content.append(meta);
      row.append(content);
      return row;
    });
    $("transcriptLines").replaceChildren(...(nodes.length ? nodes : [el("p", "inline-empty", "尚无转写证据。任务完成后，已保存的转写会显示在这里。") ]));
    const answer = currentAnswer();
    const transcript = answer && answer.result && answer.result.transcript;
    const issues = transcript && Array.isArray(transcript.issues) ? transcript.issues : [];
    const issueNodes = issues.map((issue) => el("p", "", text(issue)));
    if (transcript && transcript.is_blank === true) issueNodes.unshift(el("p", "", "转写结果标记为未作答。"));
    $("transcriptIssues").replaceChildren(...issueNodes);
    $("transcriptIssues").hidden = !issueNodes.length;
  }

  function focusEvidence(identifier) {
    const lines = transcriptLines();
    const index = lines.findIndex((line) => String(line.id) === String(identifier));
    if (index < 0) {
      showAlert(`证据 ${text(identifier)} 不在已保存的转写中，无法定位。`);
      return;
    }
    state.highlightedLine = lines[index].id;
    $("transcriptDetails").open = true;
    for (const [rowIndex, row] of [...$("transcriptLines").children].entries()) row.classList.toggle("is-highlighted", rowIndex === index);
    if (lines[index].image_id) showImage(lines[index].image_id);
    const row = $(`evidence-line-${index}`);
    if (row) {
      requestAnimationFrame(() => {
        row.scrollIntoView({ block: "nearest", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        row.focus({ preventScroll: true });
      });
    }
  }

  function offlineRecord() {
    const evaluation = state.evaluation;
    const records = evaluation && evaluation.available && evaluation.metrics && evaluation.metrics.records;
    return Array.isArray(records) ? records.find((record) => String(record.candidate_id) === String(state.candidateId) && String(record.question_id) === state.questionId) || null : null;
  }

  function renderScoring() {
    const answer = currentAnswer();
    const result = answer && answer.result;
    const grade = result && result.grade;
    const rubric = currentRubric();
    const question = currentQuestion();
    const maximum = rubric ? rubric.max_score : question && question.max_score;
    const answerState = answer ? answer.state : "missing";
    setTag($("answerStatus"), answerState);
    $("scoreValue").textContent = grade ? number(grade.total_score) : "—";
    $("scoreMax").textContent = number(maximum);
    const scoreSummary = grade && grade.status === "ungradable" ? "本题无法给出可靠分数"
      : answerState === "failed" ? "任务失败，未产生有效得分"
      : !grade ? (answer ? "等待评分结果" : "本考生未提交此题") : "按固定评分标准计算";
    $("scoreSummary").textContent = scoreSummary;
    const verification = $("verificationMeta");
    verification.replaceChildren();
    if (result && result.verified === true) verification.append(icon("check"), document.createTextNode("已运行复核"));
    else verification.textContent = result && result.verified === false ? "未运行复核" : "复核状态未记录";
    const reference = offlineRecord();
    $("referenceScore").textContent = reference ? number(reference.label_score) : "—";
    $("absoluteError").textContent = reference ? number(reference.absolute_error) : "—";
    $("offlineAnswerNote").textContent = reference ? "仅用于事后评估" : "无匹配的离线参考";

    const issues = [];
    if (answer && answer.error) issues.push(`执行错误：${text(answer.error)}`);
    if (grade && Array.isArray(grade.issues)) issues.push(...grade.issues.map((issue) => text(issue)));
    if (grade && grade.status === "ungradable" && !issues.length) issues.push("当前作答无法可靠评分，得分保持未知，请结合作答图片与证据查看。 ");
    $("answerIssues").replaceChildren(...issues.map((issue) => el("p", "", issue)));
    $("answerIssues").hidden = !issues.length;
    $("answerIssues").classList.toggle("is-error", answerState === "failed");

    const definitions = rubric && Array.isArray(rubric.criteria) ? rubric.criteria : [];
    const scores = grade && Array.isArray(grade.scores) ? grade.scores : [];
    const allCriteria = definitions.map((definition) => ({
      definition, score: scores.find((score) => score.criterion_id === definition.id) || null,
    }));
    for (const score of scores) {
      if (!definitions.some((definition) => definition.id === score.criterion_id)) allCriteria.push({ definition: null, score });
    }
    const availableIds = new Set(transcriptLines().map((line) => String(line.id)));
    const cards = allCriteria.map(({ definition, score }, index) => {
      const earned = numeric(score && score.score);
      const full = numeric(definition && definition.max_score);
      const className = earned === null ? "unknown" : full !== null && earned < full ? "partial" : "";
      const card = el("article", `criterion-card ${className}`);
      const top = el("div", "criterion-top");
      const title = el("h3", "criterion-title");
      title.append(el("span", "criterion-index", String(index + 1).padStart(2, "0")));
      title.append(el("span", "", definition ? text(definition.description) : `评分项 ${text(score && score.criterion_id)}`));
      const scoreNode = el("span", "criterion-score", earned === null ? "—" : number(earned));
      scoreNode.append(el("small", "", ` / ${number(full)}`));
      top.append(title, scoreNode);
      card.append(top);
      const reason = score ? text(score.reason, "未提供判断理由") : "当前评分项尚未产生判断依据。";
      card.append(el("p", "criterion-reason", reason));
      const evidenceIds = score && Array.isArray(score.evidence_ids) ? score.evidence_ids : [];
      if (score) {
        const evidence = el("div", "evidence-chips");
        evidence.append(el("span", "evidence-label", evidenceIds.length ? "证据" : "未引用转写证据"));
        for (const identifier of evidenceIds) {
          const chip = el("button", "evidence-chip");
          chip.type = "button";
          chip.append(icon("link"), document.createTextNode(text(identifier)));
          chip.disabled = !availableIds.has(String(identifier));
          chip.title = chip.disabled ? "该证据 ID 不在已保存的转写中" : `定位转写证据 ${text(identifier)}`;
          chip.addEventListener("click", () => focusEvidence(identifier));
          evidence.append(chip);
        }
        card.append(evidence);
      }
      const details = el("details", "criterion-rules");
      const summary = el("summary", "", "固定评分标准");
      summary.append(icon("chevron"));
      const rules = el("div");
      rules.append(el("p", "", definition ? text(definition.rules, "未提供此评分项的规则") : "此记录未保存对应评分标准。"));
      if (definition && Array.isArray(definition.alternatives) && definition.alternatives.length) {
        rules.append(el("p", "", `可接受解法：${definition.alternatives.map((item) => text(item)).join("；")}`));
      }
      details.append(summary, rules);
      card.append(details);
      return card;
    });
    $("criteriaList").replaceChildren(...(cards.length ? cards : [el("p", "inline-empty", "此题暂无已保存的评分标准与逐项结果。") ]));
    renderRubric(rubric);
  }

  function renderRubric(rubric) {
    const content = $("rubricContent");
    const nodes = [el("p", "", "评分项采用本试卷已生成的固定标准；模型依据作答转写逐项判断，再汇总本题得分。没有有效分数的项目保持未知，不以 0 分代替。")];
    if (rubric && rubric.solution) nodes.push(el("h3", "", "标准解题过程"), el("p", "", text(rubric.solution)));
    if (rubric && Array.isArray(rubric.corrections) && rubric.corrections.length) {
      nodes.push(el("h3", "", "评分标准修订记录"));
      const list = el("ul");
      for (const correction of rubric.corrections) list.append(el("li", "", text(correction)));
      nodes.push(list);
    }
    content.replaceChildren(...nodes);
  }

  function renderEvaluation() {
    const evaluation = state.evaluation;
    const content = $("evaluationContent");
    if (!evaluation || !evaluation.available) {
      const reason = evaluation ? text(evaluation.reason, "当前批次没有匹配的离线参考标签。") : "当前批次尚无可用离线评估。";
      const nodes = [el("p", "evaluation-reason", reason)];
      if (evaluation && (evaluation.matched_count !== undefined || evaluation.expected_count !== undefined)) {
        nodes.push(el("p", "evaluation-reason", `匹配 ${number(evaluation.matched_count, 0)} / 预期 ${number(evaluation.expected_count, 0)} 条记录`));
      }
      content.replaceChildren(...nodes);
      return;
    }
    const metrics = evaluation.metrics || {};
    const cards = [
      ["全部预期 · ±1 分一致率", percent(evaluation.all_expected_within_1_rate), `分母：全部预期 ${number(metrics.expected_count, 0)} 题`],
      ["已评分 · 平均绝对误差", number(metrics.mae), `分母：已评分 ${number(metrics.resolved_count, 0)} 题`],
      ["有效评分覆盖率", percent(metrics.coverage), `${number(metrics.resolved_count, 0)} / ${number(metrics.expected_count, 0)} 题`],
      ["已评分 · ±1 分一致率", percent(metrics.within_1_rate), `分母：已评分 ${number(metrics.resolved_count, 0)} 题`],
    ];
    const stats = el("div", "evaluation-stats");
    for (const [label, value, note] of cards) {
      const card = el("div", "evaluation-stat");
      card.append(el("span", "", label), el("strong", "", value), el("small", "", note));
      stats.append(card);
    }
    content.replaceChildren(stats);
  }

  function renderSelectedAnswer() {
    renderQuestion();
    renderSource();
    renderTranscript();
    renderScoring();
  }

  function renderWorkspace() {
    renderJobs();
    renderHeading();
    renderMetrics();
    renderSelectors();
    renderSelectedAnswer();
    renderEvaluation();
    $("loadingState").hidden = true;
    $("emptyState").hidden = true;
    $("workspace").hidden = false;
    requestAnimationFrame(updateImageSize);
  }

  function stopPolling() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (state.pollRequest) state.pollRequest.controller.abort();
    state.pollRequest = null;
  }

  function schedulePoll() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (!state.summary || TERMINAL_JOBS.has(state.summary.state) || state.busy || state.pollPausedError || document.hidden) return;
    state.pollTimer = setTimeout(pollCurrentJob, 2000);
  }

  async function pollCurrentJob() {
    state.pollTimer = null;
    if (state.pollRequest || !state.jobId || document.hidden || state.busy) return;
    const jobId = state.jobId;
    const sequence = state.viewSequence;
    const controller = new AbortController();
    const poll = { controller, jobId };
    state.pollRequest = poll;
    try {
      const results = await Promise.allSettled([
        request(`/v1/jobs/${encodeURIComponent(jobId)}`, { signal: controller.signal }),
        request(`/v1/jobs/${encodeURIComponent(jobId)}/results?offset=${state.offset}&limit=${PAGE_SIZE}`, { signal: controller.signal }),
      ]);
      if (state.pollRequest !== poll || sequence !== state.viewSequence) return;
      const failure = results.find((result) => result.status === "rejected");
      if (failure) throw failure.reason;
      const [summary, candidates] = results.map((result) => result.value);
      if (!Array.isArray(candidates) || !summary || typeof summary !== "object") throw new APIError("运行记录格式异常，请重新读取。", null);
      state.summary = summary;
      state.candidates = candidates;
      const job = currentJob();
      if (job) { job.state = summary.state; job.counts = summary.counts; }
      if (TERMINAL_JOBS.has(summary.state)) {
        try {
          const evaluation = await request(`/v1/jobs/${encodeURIComponent(jobId)}/evaluation`, { signal: controller.signal });
          if (state.pollRequest !== poll || sequence !== state.viewSequence) return;
          state.evaluation = evaluation;
        } catch (error) {
          if (aborted(error)) return;
          state.evaluation = { available: false, reason: `离线评估读取失败：${error.message}` };
          showAlert(`批改结果已更新；离线评估读取失败：${error.message}`, error);
        }
      }
      renderWorkspace();
      connection("connected", TERMINAL_JOBS.has(summary.state) ? "已连接 · 本地服务" : "已连接 · 自动读取进度");
    } catch (error) {
      if (aborted(error) || state.pollRequest !== poll || sequence !== state.viewSequence) return;
      state.pollPausedError = true;
      connection("error", "进度读取已中断");
      showAlert(`自动读取进度已暂停：${error.message} 请点击右上角刷新，任务在服务端继续运行。`, error);
    } finally {
      if (state.pollRequest === poll) { state.pollRequest = null; schedulePoll(); }
    }
  }

  async function loadJob(jobId, { offset = 0, fromWorkbench = false } = {}) {
    if (!fromWorkbench && state.metaController) {
      state.metaController.abort();
      state.metaSequence += 1;
    }
    stopPolling();
    if (state.viewController) state.viewController.abort();
    const controller = new AbortController();
    state.viewController = controller;
    const sequence = ++state.viewSequence;
    const switching = jobId !== state.jobId;
    if (switching) {
      clearImageCache();
      state.candidateId = null;
      state.questionId = null;
      state.sourceKey = null;
      $("transcriptDetails").open = false;
      $("questionDetails").open = false;
      $("evaluationDetails").open = false;
    }
    else clearMaterialImages();
    state.jobId = jobId;
    state.offset = offset;
    state.pollPausedError = false;
    clearAlert();
    setBusy(true);
    showLoading();
    renderJobs();
    const job = currentJob();
    const examId = job && job.exam_id;
    try {
      if (!examId) throw new APIError("批次未提供关联试卷编号，无法读取评分标准。", null);
      const results = await Promise.allSettled([
        request(`/v1/jobs/${encodeURIComponent(jobId)}`, { signal: controller.signal }),
        request(`/v1/exams/${encodeURIComponent(examId)}`, { signal: controller.signal }),
        request(`/v1/jobs/${encodeURIComponent(jobId)}/results?offset=${offset}&limit=${PAGE_SIZE}`, { signal: controller.signal }),
        request(`/v1/jobs/${encodeURIComponent(jobId)}/evaluation`, { signal: controller.signal }),
      ]);
      if (sequence !== state.viewSequence) return;
      for (let index = 0; index < 3; index += 1) if (results[index].status === "rejected") throw results[index].reason;
      state.summary = results[0].value;
      state.exam = results[1].value;
      state.candidates = results[2].value;
      if (!state.summary || !state.exam || !Array.isArray(state.candidates)) throw new APIError("运行记录格式异常，无法显示。", null);
      if (results[3].status === "fulfilled") state.evaluation = results[3].value;
      else {
        const error = results[3].reason;
        if (aborted(error)) return;
        state.evaluation = { available: false, reason: `离线评估读取失败：${error.message}` };
        showAlert(`评分记录已读取；离线评估读取失败：${error.message}`, error);
      }
      if (job) { job.state = state.summary.state; job.counts = state.summary.counts; }
      renderWorkspace();
      connection("connected", TERMINAL_JOBS.has(state.summary.state) ? "已连接 · 本地服务" : "已连接 · 自动读取进度");
    } catch (error) {
      if (aborted(error) || sequence !== state.viewSequence) return;
      state.summary = null;
      state.pollPausedError = true;
      connection("error", error.status === 401 ? "需要服务凭据" : "读取失败");
      showEmpty("无法读取这次批改", error.message);
      showAlert(`运行记录读取失败：${error.message}`, error);
    } finally {
      if (sequence === state.viewSequence) { setBusy(false); schedulePoll(); }
    }
  }

  async function loadWorkbench(preferredJobId = null) {
    stopPolling();
    if (state.metaController) state.metaController.abort();
    if (state.viewController) state.viewController.abort();
    state.viewSequence += 1;
    const controller = new AbortController();
    state.metaController = controller;
    const sequence = ++state.metaSequence;
    setBusy(true);
    clearAlert();
    connection("", "正在读取");
    if (!state.jobs.length) showLoading();
    try {
      const data = await request("/v1/workbench", { signal: controller.signal });
      if (sequence !== state.metaSequence) return;
      if (!data || !Array.isArray(data.jobs) || !Array.isArray(data.exams)) throw new APIError("工作台返回格式异常，缺少批次或试卷列表。", null);
      state.jobs = data.jobs;
      state.exams = data.exams;
      const target = state.jobs.find((job) => job.id === preferredJobId)
        || state.jobs.find((job) => job.id === state.jobId) || state.jobs[0];
      renderJobs();
      if (target) await loadJob(target.id, { offset: target.id === state.jobId ? state.offset : 0, fromWorkbench: true });
      else {
        state.jobId = null;
        state.summary = null;
        state.exam = null;
        state.candidates = [];
        clearImageCache();
        $("examTitle").textContent = "主观题评分工作台";
        $("batchDescription").textContent = "从原始作答到逐项评分，每一分都有依据。";
        const ready = state.exams.some((exam) => exam.state === "ready");
        showEmpty("还没有批改记录", ready ? "已有准备好的试卷。点击“新建单题批改”，提交一份作答即可开始。" : "当前服务还没有准备完成的试卷。请先通过 API 或命令行准备试卷与评分标准，再回到这里读取。 ");
        connection("connected", "已连接 · 本地服务");
      }
    } catch (error) {
      if (aborted(error) || sequence !== state.metaSequence) return;
      state.pollPausedError = true;
      connection("error", error.status === 401 ? "需要服务凭据" : "连接失败");
      if (!state.jobs.length) showEmpty("暂时无法读取运行记录", error.message);
      showAlert(`工作台读取失败：${error.message}`, error);
    } finally {
      if (sequence === state.metaSequence) { setBusy(false); schedulePoll(); }
    }
  }

  async function exportResults() {
    if (!state.jobId) return;
    const button = $("exportButton");
    button.disabled = true;
    const jobId = state.jobId;
    try {
      const blob = await request(`/v1/jobs/${encodeURIComponent(jobId)}/export`, { responseType: "blob" });
      const url = URL.createObjectURL(blob);
      const link = el("a");
      link.href = url;
      link.download = `${String(jobId).replace(/[^a-zA-Z0-9_-]/g, "_")}.jsonl`;
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showAlert(`原始记录导出失败：${error.message}`, error); }
    finally { button.disabled = false; }
  }

  function updateLiveQuestions() {
    const exam = state.exams.find((item) => item.id === $("liveExam").value);
    const questions = examQuestions(exam);
    const options = questions.map((question) => {
      const option = el("option", "", `${questionName(question.id)} · 满分 ${number(question.max_score)}`);
      option.value = question.id;
      return option;
    });
    if (!options.length) {
      const option = el("option", "", "暂无可用题目");
      option.value = "";
      options.push(option);
    }
    $("liveQuestion").replaceChildren(...options);
    if (state.questionId && questions.some((question) => String(question.id) === state.questionId)) $("liveQuestion").value = state.questionId;
    $("startGradingButton").disabled = state.submitting || !questions.length || state.submissionUncertain;
    $("examBudgetNote").textContent = exam && numeric(exam.budget_cny) !== null
      ? `试卷共享预算 ${money(exam.budget_cny)} · 已预留 ${money(exam.committed_cost_cny)}（当前快照）`
      : "试卷未设置共享预算；本次仍使用上述批次预算上限。";
  }

  function updateBudgetPreview() {
    const budget = numeric($("liveBudget").value);
    $("budgetPreview").textContent = budget !== null && budget > 0 ? money(budget) : "请输入有效预算";
  }

  function openGradingDialog() {
    if (state.submitting) return;
    const ready = state.exams.filter((exam) => exam.state === "ready" && examQuestions(exam).length);
    $("gradingForm").reset();
    state.submissionUncertain = false;
    $("gradingFields").disabled = false;
    $("liveError").hidden = true;
    $("liveProgress").hidden = true;
    $("fileLabel").textContent = "选择一份作答文件";
    $("startGradingButton").querySelector("span").textContent = "开始评分";
    $("closeDialogButton").disabled = false;
    $("cancelDialogButton").disabled = false;
    $("cancelDialogButton").textContent = "取消";
    const options = ready.map((exam) => {
      const option = el("option", "", text(exam.title, "未命名试卷"));
      option.value = exam.id;
      return option;
    });
    if (!options.length) {
      const option = el("option", "", "暂无准备完成的试卷");
      option.value = "";
      options.push(option);
    }
    $("liveExam").replaceChildren(...options);
    const job = currentJob();
    if (job && ready.some((exam) => exam.id === job.exam_id)) $("liveExam").value = job.exam_id;
    $("liveCandidate").value = `demo-${new Date().toISOString().replace(/\D/g, "").slice(0, 14)}`;
    updateLiveQuestions();
    updateBudgetPreview();
    if (!ready.length) {
      $("liveError").textContent = "当前没有准备完成的试卷，请先通过 API 或命令行创建试卷并等待评分标准准备完成。";
      $("liveError").hidden = false;
    }
    $("gradingDialog").showModal();
  }

  function closeGradingDialog() {
    if (state.submitting) return;
    const uncertain = state.submissionUncertain;
    $("gradingDialog").close();
    if (uncertain) loadWorkbench();
  }

  function idempotencyKey() {
    if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(20));
    return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  async function startGrading(event) {
    event.preventDefault();
    if (state.submitting || state.submissionUncertain || !$("gradingForm").reportValidity()) return;
    const examId = $("liveExam").value;
    const questionId = $("liveQuestion").value;
    const candidateId = $("liveCandidate").value.trim();
    const budget = $("liveBudget").value.trim();
    const file = $("liveFile").files[0];
    const exam = state.exams.find((item) => item.id === examId && item.state === "ready");
    if (!exam || !examQuestions(exam).some((question) => String(question.id) === questionId) || !candidateId || !file || !(Number(budget) > 0)) {
      $("liveError").textContent = "请选择已就绪的试卷、有效题目、考生编号、作答文件与预算。";
      $("liveError").hidden = false;
      return;
    }
    state.submitting = true;
    $("gradingFields").disabled = true;
    $("closeDialogButton").disabled = true;
    $("cancelDialogButton").disabled = true;
    $("startGradingButton").disabled = true;
    $("startGradingButton").querySelector("span").textContent = "正在提交…";
    $("newJobButton").disabled = true;
    $("liveError").hidden = true;
    $("liveProgress").hidden = false;
    $("liveProgress").textContent = "1 / 2 · 正在上传作答文件";
    let jobRequestStarted = false;
    let created = null;
    try {
      const body = new FormData();
      body.append("file", file);
      const assets = await request("/v1/assets", { method: "POST", body });
      if (!Array.isArray(assets) || !assets.length || assets.some((asset) => !asset.id)) throw new APIError("上传未返回有效图片编号，尚未创建评分任务。", null);
      $("liveProgress").textContent = "2 / 2 · 正在创建评分批次";
      const payload = {
        exam_id: examId, question_ids: [questionId], strategy: "B", budget_cny: budget,
        candidates: [{ candidate_id: candidateId, kind: "question", answers: [{ question_id: questionId, assets: assets.map((asset) => asset.id) }] }],
      };
      jobRequestStarted = true;
      created = await request("/v1/jobs", {
        method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotencyKey() },
        body: JSON.stringify(payload),
      });
      if (!created || !created.id) throw new APIError("服务未返回批次编号，提交状态无法确认。", null);
      // Open the accepted batch even if a later workbench refresh cannot be read.
      if (!state.jobs.some((job) => job.id === created.id)) state.jobs.unshift({
        id: created.id, exam_id: examId, exam_title: exam.title, created_at: null,
        candidate_count: 1, answer_count: 1, state: created.state, counts: created.counts,
      });
    } catch (error) {
      const ambiguous = jobRequestStarted && (!error.status || error.status >= 500 || (error.status >= 200 && error.status < 300));
      state.submissionUncertain = ambiguous;
      $("liveError").textContent = ambiguous
        ? `提交状态待确认：${error.message} 服务可能已创建任务，请先读取运行记录核对，避免再次提交。`
        : `提交失败：${error.message}${!jobRequestStarted ? " 尚未创建评分任务。" : ""}`;
      $("liveError").hidden = false;
      if (error.status === 401) showAlert("提交批改需要有效的服务访问凭据。关闭窗口后可填写凭据。", error);
    } finally {
      state.submitting = false;
      $("liveProgress").hidden = true;
      $("gradingFields").disabled = false;
      $("closeDialogButton").disabled = false;
      $("cancelDialogButton").disabled = false;
      $("cancelDialogButton").textContent = state.submissionUncertain ? "关闭并读取记录" : "取消";
      $("startGradingButton").querySelector("span").textContent = state.submissionUncertain ? "提交状态待确认" : "开始评分";
      $("startGradingButton").disabled = state.submissionUncertain;
      $("newJobButton").disabled = state.busy;
    }
    if (created && created.id) {
      $("gradingDialog").close();
      await loadJob(created.id, { offset: 0 });
    }
  }

  $("serviceToken").value = state.token;
  $("authForm").addEventListener("submit", (event) => {
    event.preventDefault();
    state.token = $("serviceToken").value.trim();
    try {
      if (state.token) sessionStorage.setItem("ai4edu-service-token", state.token);
      else sessionStorage.removeItem("ai4edu-service-token");
    } catch (_) { /* The in-memory credential still works if storage is unavailable. */ }
    clearImageCache();
    loadWorkbench();
  });
  $("alertAuthButton").addEventListener("click", focusAuth);
  $("dismissAlert").addEventListener("click", clearAlert);
  $("refreshButton").addEventListener("click", () => loadWorkbench());
  $("emptyRefreshButton").addEventListener("click", () => loadWorkbench());
  $("candidateSelect").addEventListener("change", () => {
    state.candidateId = $("candidateSelect").value;
    state.highlightedLine = null;
    renderSelectors();
    renderSelectedAnswer();
  });
  $("questionSelect").addEventListener("change", () => {
    state.questionId = $("questionSelect").value;
    state.highlightedLine = null;
    renderSelectedAnswer();
  });
  $("previousPage").addEventListener("click", () => loadJob(state.jobId, { offset: Math.max(0, state.offset - PAGE_SIZE) }));
  $("nextPage").addEventListener("click", () => loadJob(state.jobId, { offset: state.offset + PAGE_SIZE }));
  $("evaluationMetric").addEventListener("click", () => {
    $("evaluationDetails").open = true;
    $("evaluationDetails").scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  $("zoomIn").addEventListener("click", () => { state.zoom = Math.min(3, state.zoom + 0.25); updateImageSize(); });
  $("zoomOut").addEventListener("click", () => { state.zoom = Math.max(1, state.zoom - 0.25); updateImageSize(); });
  $("zoomReset").addEventListener("click", () => { state.zoom = 1; updateImageSize(); });
  $("openImageButton").addEventListener("click", () => {
    if (state.activeImageUrl) window.open(state.activeImageUrl, "_blank", "noopener,noreferrer");
  });
  $("exportButton").addEventListener("click", exportResults);
  $("newJobButton").addEventListener("click", openGradingDialog);
  $("closeDialogButton").addEventListener("click", closeGradingDialog);
  $("cancelDialogButton").addEventListener("click", closeGradingDialog);
  $("gradingDialog").addEventListener("cancel", (event) => {
    event.preventDefault();
    if (!state.submitting) closeGradingDialog();
  });
  $("gradingDialog").addEventListener("click", (event) => {
    if (event.target !== $("gradingDialog") || state.submitting) return;
    const bounds = $("gradingDialog").getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeGradingDialog();
  });
  $("liveExam").addEventListener("change", updateLiveQuestions);
  $("liveBudget").addEventListener("input", updateBudgetPreview);
  $("liveFile").addEventListener("change", () => {
    const file = $("liveFile").files[0];
    $("fileLabel").textContent = file ? `${file.name} · ${number(file.size / 1024 / 1024, 2)} MB` : "选择一份作答文件";
  });
  $("gradingForm").addEventListener("submit", startGrading);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPolling(); else schedulePoll(); });
  window.addEventListener("resize", updateImageSize);
  window.addEventListener("pagehide", () => { stopPolling(); clearImageCache(); });
  window.addEventListener("pageshow", (event) => { if (event.persisted) loadWorkbench(); });
  loadWorkbench();
})();
