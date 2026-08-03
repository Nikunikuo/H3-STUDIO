const $ = (selector) => document.querySelector(selector);
const localMutationHeaders = Object.freeze({ "X-H3-Studio-Request": "1" });
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const ui = {
  modelDot: $("#model-dot"),
  modelStatus: $("#model-status"),
  gpuStatus: $("#gpu-status"),
  ramStatus: $("#ram-status"),
  modeGrid: $("#mode-grid"),
  modeDescription: $("#mode-description"),
  omniLock: $("#omni-lock"),
  attachmentSection: $("#attachment-section"),
  frameUploads: $("#frame-uploads"),
  omniUpload: $("#omni-upload"),
  lastUploadCard: $("#last-upload-card"),
  lastImageHint: $("#last-image-hint"),
  firstInput: $("#first-image"),
  lastInput: $("#last-image"),
  firstPreview: $("#first-preview"),
  lastPreview: $("#last-preview"),
  removeFirst: $("#remove-first"),
  removeLast: $("#remove-last"),
  referenceInput: $("#reference-files"),
  referenceDropZone: $("#reference-drop-zone"),
  referenceList: $("#reference-list"),
  refImageSize: $("#ref-image-size"),
  refImageSizeNote: $("#ref-image-size-note"),
  styleStep: $("#style-step"),
  promptStep: $("#prompt-step"),
  soundStep: $("#sound-step"),
  settingsStep: $("#settings-step"),
  prompt: $("#prompt"),
  promptCount: $("#prompt-count"),
  promptHistory: $("#prompt-history"),
  reusePrompt: $("#reuse-prompt"),
  soundSection: $("#sound-section"),
  soundStatus: $("#sound-status"),
  soundPreset: $("#sound-preset"),
  musicPolicy: $("#music-policy"),
  dialogue: $("#dialogue"),
  dialogueCount: $("#dialogue-count"),
  soundscape: $("#soundscape"),
  soundscapeCount: $("#soundscape-count"),
  audioGain: $("#audio-gain"),
  quality: $("#quality"),
  resolution: $("#resolution"),
  duration: $("#duration"),
  seed: $("#seed"),
  acceleration: $("#acceleration"),
  accelerationNote: $("#acceleration-note"),
  weightWarning: $("#weight-warning"),
  weightWarningTitle: $("#weight-warning-title"),
  weightWarningText: $("#weight-warning-text"),
  quickPreview: $("#quick-preview"),
  generate: $("#generate"),
  generateSummary: $("#generate-summary"),
  formError: $("#form-error"),
  queueBadge: $("#queue-badge"),
  emptyStage: $("#empty-stage"),
  progressStage: $("#progress-stage"),
  progressRing: $("#progress-ring"),
  progressNumber: $("#progress-number"),
  progressCaption: $("#progress-caption"),
  progressPhase: $("#progress-phase"),
  progressMessage: $("#progress-message"),
  progressElapsed: $("#progress-elapsed"),
  progressStep: $("#progress-step"),
  cancelJob: $("#cancel-job"),
  resultStage: $("#result-stage"),
  resultVideo: $("#result-video"),
  resultTitle: $("#result-title"),
  resultMeta: $("#result-meta"),
  downloadResult: $("#download-result"),
  reuseJob: $("#reuse-job"),
  reuseErrorPrompt: $("#reuse-error-prompt"),
  errorStage: $("#error-stage"),
  errorTitle: $("#error-title"),
  errorMessage: $("#error-message"),
  showLog: $("#show-log"),
  logView: $("#log-view"),
  historyGrid: $("#history-grid"),
  toast: $("#toast"),
};

const state = {
  mode: "t2v",
  style: "natural",
  firstFile: null,
  lastFile: null,
  references: [],
  capabilities: null,
  jobs: [],
  selectedJobId: null,
  pollTimer: null,
  toastTimer: null,
  promptHistorySignature: "",
  previewUrls: { first: null, last: null },
};

const modeDescriptions = {
  t2v: "テキストだけで、映像と同期音声を同時に生成します。",
  i2v: "1枚の画像を開始フレームにして、動きと同期音声を生成します。",
  first_last: "開始と終了の2枚を指定して、その間の動きと音を生成します。",
  omni: "画像・動画・音声を順番付きで参照します。",
};

function updateModeDescription() {
  if (state.mode !== "omni") {
    ui.modeDescription.textContent = modeDescriptions[state.mode];
    return;
  }
  const useMax = ui.refImageSize.value === "max";
  ui.modeDescription.textContent = useMax
    ? "画像・動画・音声を順番付きで参照します。画像はアップスケールせず、短辺2048pxを上限に高精度で解析します。"
    : "画像・動画・音声を順番付きで参照します。画像は出力面積に合わせて縮小し、軽い設定で解析します。";
  ui.refImageSizeNote.textContent = useMax
    ? "画像はアップスケールせず、短辺2048pxを上限に元解像度寄りで解析します。細部を残しやすい一方、Qwen解析とattentionが重くなります。"
    : "画像を出力面積に合わせて縮小して解析します。通常はこちらが速く、参照処理の負荷も抑えられます。";
  ui.refImageSizeNote.classList.toggle("high-precision", useMax);
}

const statusLabels = {
  queued: "待機中",
  running: "生成中",
  completed: "完成",
  failed: "失敗",
  cancelled: "中止",
  interrupted: "中断",
};

const modeLabels = {
  t2v: "Text",
  i2v: "Image",
  first_last: "Frames",
  omni: "Omni",
};

const audioPresetLabels = {
  auto: "音は自動",
  dialogue: "会話優先",
  ambience: "環境音優先",
  effects: "効果音優先",
  music: "音楽優先",
  quiet: "静かな音場",
};

const accelerationLabels = {
  off: "高速化OFF",
  conservative: "EasyCache 保守的",
  balanced: "EasyCache 高速",
};

function jobAccelerationMeta(job) {
  const mode = job.acceleration || job.acceleration_mode || "off";
  const cache = job.cache || job.easycache || job.easy_cache || job.cache_stats || {};
  const skipped = cache.skipped_steps
    ?? cache.skipped
    ?? job.easycache_skipped_steps
    ?? job.cache_skipped_steps;
  const total = cache.total_steps ?? cache.total ?? job.easycache_total_steps;
  const backend = job.backend || job.engine_backend;
  const parts = [];
  if (backend) parts.push(`Backend ${backend}`);
  if (job.attention_backend === "sage") parts.push("SageAttention");
  if (mode !== "off") {
    if (cache.enabled === false) {
      parts.push(cache.reason === "steps_below_12" ? "EasyCache 自動OFF（Draft）" : "EasyCache OFF");
    } else {
      const threshold = Number.isFinite(Number(cache.reuse_threshold))
        ? Number(cache.reuse_threshold).toFixed(2)
        : mode === "conservative" ? "0.20" : mode === "balanced" ? "0.30" : mode;
      let label = `EasyCache ${threshold}`;
      if (Number.isFinite(Number(skipped))) {
        label += ` · ${Number(skipped)}${Number.isFinite(Number(total)) ? `/${Number(total)}` : ""} skip`;
      }
      if (Number.isFinite(Number(cache.speedup)) && Number(cache.speedup) > 0) {
        label += ` · ×${Number(cache.speedup).toFixed(2)}`;
      }
      parts.push(label);
    }
  }
  return parts;
}

const samples = {
  t2v: "A cinematic night scene in Tokyo. A black sports car moves slowly through a rain-soaked neon street. Reflections shimmer on the asphalt. Rain, tires passing through puddles, and distant city ambience can be heard.",
  i2v: "The subject begins moving naturally and looks toward the camera. The camera slowly pushes in. Subtle room ambience and soft clothing movement can be heard.",
  first_last: "A smooth cinematic transition connects the first frame to the last. Motion remains physically natural and temporally coherent, with matching ambient sound.",
  omni: "Use the references in order for subject identity, visual language, motion, and sound. Create a coherent cinematic scene with synchronized natural audio.",
};

function toast(message) {
  clearTimeout(state.toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.add("visible");
  state.toastTimer = setTimeout(() => ui.toast.classList.remove("visible"), 3300);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
}

function formatElapsed(iso) {
  if (!iso) return "経過 00:00";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  const formatted = hours
    ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`
    : `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  return `経過 ${formatted}`;
}

function setPromptValue(value, focus = false) {
  ui.prompt.value = String(value || "").slice(0, 4000);
  ui.promptCount.textContent = String(ui.prompt.value.length);
  ui.formError.textContent = "";
  if (focus) ui.prompt.focus();
}

function secondsSince(iso) {
  if (!iso) return 0;
  return Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
}

function stageEstimate(actual, ceiling, elapsed, timeConstant) {
  if (ceiling <= actual) return actual;
  const fraction = 0.92 * (1 - Math.exp(-elapsed / Math.max(timeConstant, 1)));
  return Math.min(ceiling, actual + (ceiling - actual) * fraction);
}

function denoiseStepSeconds(job) {
  const quickWork = 320 * 192 * 124;
  const work = Math.max(quickWork, Number(job.width) * Number(job.height) * Number(job.num_frames));
  const referenceFactor = job.mode === "omni" ? 1 + Math.min(4, job.attachments?.length || 0) * 0.28 : 1;
  return Math.max(5, Math.min(240, 5 * Math.pow(work / quickWork, 0.8) * referenceFactor));
}

function displayedProgress(job) {
  const actual = Math.max(0, Math.min(100, Number(job.progress) || 0));
  if (job.status !== "running" || actual >= 100) return { value: actual, estimated: false };
  const elapsed = secondsSince(job.progress_updated_at || job.started_at || job.created_at);

  if (actual >= 55 && actual < 90 && Number(job.total_steps) > 0) {
    const total = Number(job.total_steps);
    const step = Number(job.step) || 0;
    const nextBoundary = Math.min(90, 55 + 35 * (step + 1) / total - 0.04);
    const value = stageEstimate(actual, nextBoundary, elapsed, denoiseStepSeconds(job));
    return { value, estimated: value > actual + 0.04 };
  }

  let ceiling = actual;
  let timeConstant = 30;
  if (actual < 4) [ceiling, timeConstant] = [3.8, 8];
  else if (actual < 7) [ceiling, timeConstant] = [6.8, 12];
  else if (actual < 17) [ceiling, timeConstant] = [16.8, 75];
  else if (actual < 26) [ceiling, timeConstant] = [25.8, 85];
  else if (actual < 30) [ceiling, timeConstant] = [29.8, 35];
  else if (actual < 45) {
    ceiling = 44.8;
    timeConstant = job.mode === "omni" ? 180 * Math.max(1, job.attachments?.length || 1) : 55;
  } else if (actual < 52) {
    ceiling = 51.8;
    timeConstant = job.mode === "omni" ? 150 * Math.max(1, job.attachments?.length || 1) : 70;
  } else if (actual < 55) [ceiling, timeConstant] = [54.8, 45];
  else if (actual < 91) [ceiling, timeConstant] = [90.8, 40];
  else if (actual < 92) [ceiling, timeConstant] = [91.8, 90];
  else if (actual < 93) [ceiling, timeConstant] = [92.8, 45];
  else if (actual < 96) [ceiling, timeConstant] = [95.8, 35];
  else if (actual < 100) [ceiling, timeConstant] = [99.4, 22];

  const value = stageEstimate(actual, ceiling, elapsed, timeConstant);
  return { value, estimated: value > actual + 0.04 };
}

function selectedJob() {
  return state.jobs.find((job) => job.id === state.selectedJobId) || null;
}

function setMode(mode) {
  if (mode === "omni" && state.capabilities && !state.capabilities.modes.omni) {
    toast("OmniはRef2VA公式重みの準備後に有効になります。");
    return;
  }
  state.mode = mode;
  $$(".mode-card").forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-checked", String(selected));
  });
  updateModeDescription();
  const hasAttachment = mode !== "t2v";
  ui.attachmentSection.classList.toggle("hidden", !hasAttachment);
  ui.frameUploads.classList.toggle("hidden", mode === "omni");
  ui.omniUpload.classList.toggle("hidden", mode !== "omni");
  ui.lastUploadCard.classList.toggle("hidden", mode === "i2v");
  ui.lastImageHint.textContent = mode === "first_last" ? "必須" : "任意";
  ui.styleStep.textContent = hasAttachment ? "03" : "02";
  ui.promptStep.textContent = hasAttachment ? "04" : "03";
  ui.soundStep.textContent = hasAttachment ? "05" : "04";
  ui.settingsStep.textContent = hasAttachment ? "06" : "05";
  updateSummary();
}

function setStyle(style) {
  state.style = style;
  $$(".style-chip").forEach((button) => button.classList.toggle("selected", button.dataset.style === style));
}

function setFrameFile(which, file) {
  const isFirst = which === "first";
  const stateKey = isFirst ? "firstFile" : "lastFile";
  const preview = isFirst ? ui.firstPreview : ui.lastPreview;
  const remove = isFirst ? ui.removeFirst : ui.removeLast;
  const oldUrl = state.previewUrls[which];
  if (oldUrl) URL.revokeObjectURL(oldUrl);
  state[stateKey] = file || null;
  state.previewUrls[which] = file ? URL.createObjectURL(file) : null;
  preview.replaceChildren();
  if (file) {
    const image = document.createElement("img");
    image.src = state.previewUrls[which];
    image.alt = isFirst ? "開始画像プレビュー" : "終了画像プレビュー";
    preview.append(image);
    remove.classList.remove("hidden");
  } else {
    const symbol = document.createElement("span");
    symbol.className = "upload-symbol";
    symbol.textContent = "＋";
    preview.append(symbol);
    remove.classList.add("hidden");
  }
}

function referenceType(file) {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("audio/")) return "audio";
  return "unknown";
}

function referenceKind(file) {
  return { image: "画像", video: "動画", audio: "音声" }[referenceType(file)] || "素材";
}

function referenceTag(type, index) {
  const label = { image: "Picture", video: "Video", audio: "Audio" }[type] || "Reference";
  return `<${label} ${index}>`;
}

function insertReferenceTag(tag) {
  const start = ui.prompt.selectionStart ?? ui.prompt.value.length;
  const end = ui.prompt.selectionEnd ?? start;
  const before = ui.prompt.value.slice(0, start);
  const after = ui.prompt.value.slice(end);
  const leading = before && !/\s$/.test(before) ? " " : "";
  const trailing = after && !/^\s/.test(after) ? " " : "";
  const insertion = `${leading}${tag}${trailing}`;
  const value = `${before}${insertion}${after}`.slice(0, 4000);
  setPromptValue(value, true);
  const cursor = Math.min(value.length, before.length + insertion.length - trailing.length);
  ui.prompt.setSelectionRange(cursor, cursor);
  toast(`${tag} をプロンプトへ挿入しました。`);
}

function addReferences(files) {
  const incoming = Array.from(files).filter((file) => /^(image|video|audio)\//.test(file.type));
  const remaining = Math.max(0, 12 - state.references.length);
  state.references.push(...incoming.slice(0, remaining));
  if (incoming.length > remaining) toast("Omni参照は合計12素材までです。");
  renderReferences();
}

function renderReferences() {
  ui.referenceList.replaceChildren();
  const counts = { image: 0, video: 0, audio: 0 };
  state.references.forEach((file, index) => {
    const type = referenceType(file);
    counts[type] = (counts[type] || 0) + 1;
    const tag = referenceTag(type, counts[type]);
    const item = document.createElement("div");
    item.className = "reference-item";

    const number = document.createElement("span");
    number.className = "reference-index";
    number.textContent = String(index + 1).padStart(2, "0");

    const copy = document.createElement("span");
    copy.className = "reference-name";
    const nameLine = document.createElement("span");
    nameLine.className = "reference-name-line";
    const tagButton = document.createElement("button");
    tagButton.type = "button";
    tagButton.className = "reference-tag";
    tagButton.textContent = tag;
    tagButton.title = `${tag} をプロンプトへ挿入`;
    tagButton.setAttribute("aria-label", `${tag} をプロンプトへ挿入`);
    tagButton.addEventListener("click", () => insertReferenceTag(tag));
    const name = document.createElement("strong");
    name.textContent = file.name;
    nameLine.append(tagButton, name);
    const meta = document.createElement("small");
    meta.textContent = `${referenceKind(file)} · ${formatBytes(file.size)} · UI順 ${String(index + 1).padStart(2, "0")}`;
    copy.append(nameLine, meta);

    const actions = document.createElement("span");
    actions.className = "reference-actions";
    [
      ["↑", -1, "前へ"],
      ["↓", 1, "後ろへ"],
      ["×", 0, "削除"],
    ].forEach(([label, direction, title]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.title = title;
      button.addEventListener("click", () => {
        if (direction === 0) {
          state.references.splice(index, 1);
        } else {
          const target = index + direction;
          if (target >= 0 && target < state.references.length) {
            [state.references[index], state.references[target]] = [state.references[target], state.references[index]];
          }
        }
        renderReferences();
      });
      actions.append(button);
    });
    item.append(number, copy, actions);
    ui.referenceList.append(item);
  });
  updateSummary();
}

function renderPromptHistory() {
  const previous = ui.promptHistory.value;
  const unique = [];
  const prompts = new Set();
  state.jobs.forEach((job) => {
    const normalized = String(job.prompt || "").trim();
    const promptKey = [
      normalized,
      job.audio_preset || "auto",
      job.dialogue || "",
      job.soundscape || "",
      job.music_policy || "auto",
      job.audio_gain_db ?? 0,
    ].join("\u0001");
    if (!normalized || prompts.has(promptKey)) return;
    prompts.add(promptKey);
    unique.push(job);
  });
  const signature = unique.slice(0, 20).map((job) => [
    job.id,
    job.prompt,
    job.audio_preset,
    job.dialogue,
    job.soundscape,
    job.music_policy,
    job.audio_gain_db,
  ].join("\u0001")).join("\u0000");
  if (signature === state.promptHistorySignature) return;
  state.promptHistorySignature = signature;

  ui.promptHistory.replaceChildren();
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = unique.length ? "過去のプロンプトを選択…" : "再利用できるプロンプトはまだありません";
  ui.promptHistory.append(placeholder);
  unique.slice(0, 20).forEach((job) => {
    const option = document.createElement("option");
    option.value = job.id;
    const oneLine = job.prompt.replace(/\s+/g, " ").trim();
    const audioLabel = audioPresetLabels[job.audio_preset || "auto"] || "音は自動";
    option.textContent = `[${modeLabels[job.mode] || job.mode} · ${audioLabel}] ${oneLine.length > 62 ? `${oneLine.slice(0, 62)}…` : oneLine}`;
    ui.promptHistory.append(option);
  });
  if (unique.some((job) => job.id === previous)) ui.promptHistory.value = previous;
  ui.reusePrompt.disabled = !ui.promptHistory.value;
}

function restoreAudioPrompt(job) {
  ui.soundPreset.value = job?.audio_preset || "auto";
  ui.musicPolicy.value = job?.music_policy || "auto";
  ui.dialogue.value = job?.dialogue || "";
  ui.dialogueCount.textContent = String(ui.dialogue.value.length);
  ui.soundscape.value = job?.soundscape || "";
  ui.soundscapeCount.textContent = String(ui.soundscape.value.length);
  ui.audioGain.value = String(job?.audio_gain_db ?? 0);
  updateSummary();
}

function reusePromptOnly(job) {
  if (!job) return;
  setPromptValue(job.prompt, true);
  restoreAudioPrompt(job);
  ui.prompt.closest(".form-section").scrollIntoView({ behavior: "smooth", block: "center" });
  toast("プロンプトと音響指示を復元しました。参照タグもそのまま再利用できます。");
}

function updateSummary() {
  updateModeDescription();
  const quality = ui.quality.options[ui.quality.selectedIndex].text.split(" · ")[0];
  const resolution = ui.resolution.value.replace("x", "×");
  const duration = ui.duration.options[ui.duration.selectedIndex].text;
  const audio = audioPresetLabels[ui.soundPreset.value] || "音は自動";
  const requestedAcceleration = ui.acceleration.value;
  const cacheAutoOff = Number(ui.quality.value) < 12 && requestedAcceleration !== "off";
  const acceleration = cacheAutoOff
    ? "EasyCache 自動OFF"
    : accelerationLabels[requestedAcceleration] || "高速化OFF";
  ui.generateSummary.textContent = `${resolution} · ${duration} · ${quality} · ${audio} · ${acceleration}`;
  ui.accelerationNote.classList.toggle("auto-off", cacheAutoOff);
  ui.accelerationNote.textContent = cacheAutoOff
    ? "Draft（steps<12）のため、この生成ではEasyCacheを自動OFFにします。Standard / Highでは選択した近似処理を使い、映像と音声がわずかに変わる可能性があります。"
    : "EasyCacheは前ステップの計算を再利用する近似処理です。20 step前後で効果が出やすく、Draft（steps<12）では自動OFF。映像と音声がわずかに変わる可能性があります。";
  const [width, height] = ui.resolution.value.split("x").map(Number);
  const denoiseForwards = Math.max(1, Number(ui.quality.value) - 1);
  const baselineWork = 960 * 544 * 124 * 7;
  const outputWork = width * height * Number(ui.duration.value) * denoiseForwards;
  const relativeWork = outputWork / baselineWork;
  const omniReferences = state.mode === "omni" ? state.references.length : 0;
  const omniImageReferences = state.mode === "omni"
    ? state.references.filter((file) => file.type.startsWith("image/")).length
    : 0;
  const highPrecisionReferences = omniImageReferences > 0 && ui.refImageSize.value === "max";
  const heavy = relativeWork > 2 || omniReferences >= 2 || highPrecisionReferences;
  ui.weightWarning.classList.toggle("hidden", !heavy);
  if (!heavy) return;

  let severity = "高負荷設定です";
  if (relativeWork >= 6) severity = "かなり高負荷です";
  if (relativeWork >= 12 || (relativeWork >= 6 && omniReferences >= 2)) severity = "最大級の負荷です";
  ui.weightWarningTitle.textContent = `${severity}（出力側の概算 ×${relativeWork.toFixed(1)}）`;
  const referencePolicy = !omniImageReferences
    ? ""
    : ui.refImageSize.value === "max"
      ? " 高精度設定の画像はアップスケールせず、短辺2048pxを上限に解析するためmatchより重くなります。"
      : " 高速設定の画像は出力面積に合わせて縮小し、参照解析負荷を抑えます。";
  const referenceNote = omniReferences
    ? ` さらにOmni参照${omniReferences}件のQwen解析とattention負荷が加わります。${referencePolicy}`
    : "";
  ui.weightWarningText.textContent = `基準は960×544・約5秒・Draftです。これは画素数・長さ・denoise回数だけの相対目安です。${referenceNote}`;
}

function validateForm() {
  const prompt = ui.prompt.value.trim();
  if (!prompt) return "プロンプトを入力してください。";
  if (state.mode === "i2v" && !state.firstFile) return "開始画像を追加してください。";
  if (state.mode === "first_last" && (!state.firstFile || !state.lastFile)) return "開始画像と終了画像を追加してください。";
  if (state.mode === "omni" && !state.references.length) return "Omni参照素材を追加してください。";
  if (state.mode === "omni" && state.capabilities && !state.capabilities.modes.omni) return "Omniモデルはまだ準備中です。";
  return null;
}

async function submitJob() {
  const error = validateForm();
  ui.formError.textContent = error || "";
  if (error) return;

  const [width, height] = ui.resolution.value.split("x").map(Number);
  const data = new FormData();
  data.append("mode", state.mode);
  data.append("style", state.style);
  data.append("prompt", ui.prompt.value.trim());
  data.append("width", String(width));
  data.append("height", String(height));
  data.append("num_frames", ui.duration.value);
  data.append("steps", ui.quality.value);
  data.append("seed", ui.seed.value || "42");
  data.append("acceleration", ui.acceleration.value);
  data.append("ref_image_size", ui.refImageSize.value);
  data.append("audio_preset", ui.soundPreset.value);
  data.append("dialogue", ui.dialogue.value.trim());
  data.append("soundscape", ui.soundscape.value.trim());
  data.append("music_policy", ui.musicPolicy.value);
  data.append("audio_gain_db", ui.audioGain.value);
  if (["i2v", "first_last"].includes(state.mode) && state.firstFile) data.append("first_image", state.firstFile);
  if (state.mode === "first_last" && state.lastFile) data.append("last_image", state.lastFile);
  if (state.mode === "omni") state.references.forEach((file) => data.append("references", file));

  ui.generate.disabled = true;
  ui.formError.textContent = "";
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: localMutationHeaders,
      body: data,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "生成を開始できませんでした。");
    state.selectedJobId = payload.id;
    toast("生成キューへ追加しました。画面を閉じずにお待ちください。");
    await refreshState();
  } catch (errorValue) {
    ui.formError.textContent = errorValue.message;
  } finally {
    ui.generate.disabled = false;
  }
}

function updateSystem(snapshot, capabilities) {
  state.capabilities = capabilities;
  const ready = capabilities.fl2va;
  ui.modelDot.classList.toggle("ready", ready);
  ui.modelStatus.textContent = ready ? "FL2VA READY" : "MODEL MISSING";
  ui.omniLock.textContent = capabilities.ref2va ? "READY" : "Ref2VA準備中";
  ui.omniLock.classList.toggle("ready", capabilities.ref2va);
  const soundControlsReady = capabilities.audio_controls?.supported === true;
  ui.soundSection.classList.toggle("pending", !soundControlsReady);
  [ui.soundPreset, ui.musicPolicy, ui.dialogue, ui.soundscape, ui.audioGain].forEach((element) => {
    element.disabled = !soundControlsReady;
  });
  ui.soundStatus.textContent = soundControlsReady
    ? "会話・環境音・効果音・BGMの指示を、公式推奨の音響プロンプト形式へ自動でまとめます。"
    : "現在の生成が終わったあと、H3 Studioサーバーを再起動すると音響コントロールが有効になります。";

  if (snapshot.gpu) {
    ui.gpuStatus.textContent = `GPU ${snapshot.gpu.utilization}% · ${Math.round(snapshot.gpu.memory_used_mib / 1024)}GB`;
  } else {
    ui.gpuStatus.textContent = "GPU —";
  }
  ui.ramStatus.textContent = `RAM ${snapshot.ram.available_gib}GB free`;
}

function showOnly(name) {
  const mapping = {
    empty: ui.emptyStage,
    progress: ui.progressStage,
    result: ui.resultStage,
    error: ui.errorStage,
  };
  Object.entries(mapping).forEach(([key, element]) => element.classList.toggle("hidden", key !== name));
}

function renderSelectedJob() {
  const job = selectedJob();
  if (!job) {
    ui.queueBadge.textContent = "待機中";
    showOnly("empty");
    return;
  }

  ui.queueBadge.textContent = statusLabels[job.status] || job.status;
  if (["queued", "running"].includes(job.status)) {
    showOnly("progress");
    const progress = displayedProgress(job);
    ui.progressRing.style.setProperty("--progress", `${progress.value * 3.6}deg`);
    const showDecimal = progress.estimated || !Number.isInteger(Number(job.progress));
    ui.progressNumber.textContent = `${showDecimal ? progress.value.toFixed(1) : Math.round(progress.value)}%`;
    ui.progressCaption.textContent = progress.estimated ? "ESTIMATE" : "PROGRESS";
    ui.progressPhase.textContent = job.phase || "準備しています";
    ui.progressMessage.textContent = job.message || "ローカルで処理しています。";
    ui.progressElapsed.textContent = formatElapsed(job.started_at);
    ui.progressStep.textContent = job.total_steps
      ? `${job.step || 0} / ${job.total_steps} steps${progress.estimated ? " · ステップ内推定" : ""}`
      : progress.estimated ? "段階内の推定進捗" : "モデル準備中";
    ui.cancelJob.disabled = !job.can_cancel;
    return;
  }

  if (job.status === "completed") {
    showOnly("result");
    if (ui.resultVideo.dataset.jobId !== job.id) {
      ui.resultVideo.src = `${job.result}?v=${encodeURIComponent(job.finished_at || Date.now())}`;
      ui.resultVideo.poster = job.preview || "";
      ui.resultVideo.dataset.jobId = job.id;
      ui.resultVideo.load();
    }
    ui.resultTitle.textContent = job.prompt.length > 70 ? `${job.prompt.slice(0, 70)}…` : job.prompt;
    ui.resultMeta.textContent = [
      `${job.width}×${job.height}`,
      `${job.num_frames} frames`,
      `${job.steps - 1} denoise`,
      "32 kHz stereo",
      ...jobAccelerationMeta(job),
    ].join(" · ");
    ui.downloadResult.href = job.result;
    return;
  }

  showOnly("error");
  ui.errorTitle.textContent = job.status === "cancelled" ? "生成をキャンセルしました" : "生成を完了できませんでした";
  ui.errorMessage.textContent = job.message || "詳細ログを確認してください。";
  ui.reuseErrorPrompt.disabled = !job.prompt;
  ui.logView.textContent = (job.logs || []).map((entry) => entry.text).join("\n");
  ui.logView.classList.add("hidden");
}

function createHistoryCard(job) {
  const card = document.createElement("article");
  card.className = "history-card";
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `${statusLabels[job.status] || job.status}: ${job.prompt}`);

  const thumb = document.createElement("div");
  thumb.className = "history-thumb";
  if (job.preview) {
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = job.preview;
    image.alt = "生成動画のプレビュー";
    thumb.append(image);
  }
  const badge = document.createElement("span");
  badge.className = `history-status ${job.status}`;
  badge.textContent = statusLabels[job.status] || job.status;
  thumb.append(badge);

  const copy = document.createElement("div");
  copy.className = "history-copy";
  const prompt = document.createElement("strong");
  prompt.textContent = job.prompt;
  const meta = document.createElement("small");
  meta.textContent = [
    `${job.width}×${job.height}`,
    `${job.steps - 1} steps`,
    ...jobAccelerationMeta(job),
  ].join(" · ");
  copy.append(prompt, meta);
  card.append(thumb, copy);

  const select = () => {
    state.selectedJobId = job.id;
    renderSelectedJob();
    $("#stage-card").scrollIntoView({ behavior: "smooth", block: "center" });
  };
  card.addEventListener("click", select);
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select();
    }
  });
  return card;
}

function renderHistory() {
  ui.historyGrid.replaceChildren();
  if (!state.jobs.length) {
    const empty = document.createElement("div");
    empty.className = "history-empty";
    empty.textContent = "生成履歴はまだありません。";
    ui.historyGrid.append(empty);
    return;
  }
  state.jobs.slice(0, 9).forEach((job) => ui.historyGrid.append(createHistoryCard(job)));
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error("status unavailable");
    const payload = await response.json();
    state.jobs = payload.jobs;
    updateSystem(payload.system, payload.capabilities);
    if (!state.selectedJobId) {
      const active = state.jobs.find((job) => ["queued", "running"].includes(job.status));
      state.selectedJobId = active?.id || state.jobs[0]?.id || null;
    }
    renderSelectedJob();
    renderHistory();
    renderPromptHistory();
  } catch {
    ui.modelDot.classList.remove("ready");
    ui.modelStatus.textContent = "接続を確認中";
  }
}

function resetForm() {
  setMode("t2v");
  setStyle("natural");
  setPromptValue("");
  ui.quality.value = "20";
  ui.resolution.value = "640x384";
  ui.duration.value = "124";
  ui.seed.value = "42";
  ui.acceleration.value = "off";
  ui.refImageSize.value = "match";
  ui.soundPreset.value = "auto";
  ui.musicPolicy.value = "auto";
  ui.dialogue.value = "";
  ui.dialogueCount.textContent = "0";
  ui.soundscape.value = "";
  ui.soundscapeCount.textContent = "0";
  ui.audioGain.value = "0";
  setFrameFile("first", null);
  setFrameFile("last", null);
  state.references = [];
  renderReferences();
  updateSummary();
  ui.formError.textContent = "";
}

function reuseSelectedJob() {
  const job = selectedJob();
  if (!job) return;
  setMode(job.mode === "omni" && !state.capabilities?.modes.omni ? "t2v" : job.mode);
  setStyle(job.style);
  setPromptValue(job.prompt);
  ui.quality.value = String(job.steps);
  ui.resolution.value = `${job.width}x${job.height}`;
  ui.duration.value = String(job.num_frames);
  ui.seed.value = String(job.seed);
  ui.acceleration.value = job.acceleration || job.acceleration_mode || "off";
  ui.refImageSize.value = job.ref_image_size || "match";
  restoreAudioPrompt(job);
  updateSummary();
  if (job.attachments?.length) toast("設定を戻しました。参照素材だけ再添付してください。");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function cancelSelectedJob() {
  const job = selectedJob();
  if (!job || !job.can_cancel) return;
  if (!window.confirm("現在の生成をキャンセルしますか？")) return;
  const response = await fetch(`/api/jobs/${job.id}/cancel`, {
    method: "POST",
    headers: localMutationHeaders,
  });
  if (response.ok) {
    toast("生成をキャンセルしました。");
    await refreshState();
  }
}

function wireUploadDrop(label, which) {
  ["dragenter", "dragover"].forEach((name) => label.addEventListener(name, (event) => {
    event.preventDefault();
    label.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => label.addEventListener(name, (event) => {
    event.preventDefault();
    label.classList.remove("dragging");
  }));
  label.addEventListener("drop", (event) => {
    const file = Array.from(event.dataTransfer.files).find((item) => item.type.startsWith("image/"));
    if (file) setFrameFile(which, file);
  });
}

function initializeEvents() {
  ui.modeGrid.addEventListener("click", (event) => {
    const button = event.target.closest(".mode-card");
    if (button) setMode(button.dataset.mode);
  });
  $("#style-row").addEventListener("click", (event) => {
    const button = event.target.closest(".style-chip");
    if (button) setStyle(button.dataset.style);
  });
  ui.prompt.addEventListener("input", () => {
    ui.promptCount.textContent = String(ui.prompt.value.length);
    ui.formError.textContent = "";
  });
  $("#sample-prompt").addEventListener("click", () => {
    setPromptValue(samples[state.mode], true);
  });
  ui.promptHistory.addEventListener("change", () => {
    ui.reusePrompt.disabled = !ui.promptHistory.value;
  });
  ui.reusePrompt.addEventListener("click", () => {
    reusePromptOnly(state.jobs.find((job) => job.id === ui.promptHistory.value));
  });
  [ui.quality, ui.resolution, ui.duration, ui.acceleration, ui.refImageSize, ui.soundPreset, ui.musicPolicy].forEach((element) => element.addEventListener("change", updateSummary));
  ui.dialogue.addEventListener("input", () => {
    ui.dialogueCount.textContent = String(ui.dialogue.value.length);
  });
  ui.soundscape.addEventListener("input", () => {
    ui.soundscapeCount.textContent = String(ui.soundscape.value.length);
  });
  ui.quickPreview.addEventListener("click", () => {
    ui.quality.value = "8";
    ui.resolution.value = "640x384";
    ui.duration.value = "124";
    updateSummary();
    toast("軽量プレビュー設定（640×384・約5秒・Draft）へ変更しました。");
  });
  $("#random-seed").addEventListener("click", () => {
    ui.seed.value = String(Math.floor(Math.random() * 2_147_483_647));
  });
  ui.firstInput.addEventListener("change", () => setFrameFile("first", ui.firstInput.files[0] || null));
  ui.lastInput.addEventListener("change", () => setFrameFile("last", ui.lastInput.files[0] || null));
  ui.removeFirst.addEventListener("click", (event) => {
    event.preventDefault();
    setFrameFile("first", null);
    ui.firstInput.value = "";
  });
  ui.removeLast.addEventListener("click", (event) => {
    event.preventDefault();
    setFrameFile("last", null);
    ui.lastInput.value = "";
  });
  wireUploadDrop($("#first-upload-card"), "first");
  wireUploadDrop($("#last-upload-card"), "last");
  ui.referenceInput.addEventListener("change", () => {
    addReferences(ui.referenceInput.files);
    ui.referenceInput.value = "";
  });
  ["dragenter", "dragover"].forEach((name) => ui.referenceDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.referenceDropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => ui.referenceDropZone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.referenceDropZone.classList.remove("dragging");
  }));
  ui.referenceDropZone.addEventListener("drop", (event) => addReferences(event.dataTransfer.files));
  ui.generate.addEventListener("click", submitJob);
  $("#reset-form").addEventListener("click", resetForm);
  ui.cancelJob.addEventListener("click", cancelSelectedJob);
  ui.reuseJob.addEventListener("click", reuseSelectedJob);
  ui.reuseErrorPrompt.addEventListener("click", () => reusePromptOnly(selectedJob()));
  ui.showLog.addEventListener("click", () => ui.logView.classList.toggle("hidden"));
  $("#refresh-history").addEventListener("click", refreshState);
  $("#open-folder").addEventListener("click", async () => {
    const response = await fetch("/api/open-output-folder", {
      method: "POST",
      headers: localMutationHeaders,
    });
    if (!response.ok) toast("生成フォルダを開けませんでした。");
  });
}

async function initialize() {
  initializeEvents();
  setMode("t2v");
  setStyle("natural");
  updateSummary();
  await refreshState();
  state.pollTimer = setInterval(refreshState, 1500);
}

initialize();
