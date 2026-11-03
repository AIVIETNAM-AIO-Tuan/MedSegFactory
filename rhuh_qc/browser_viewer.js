(function () {
  "use strict";

  // Both uint8 volumes remain cached: at most 128 MiB of decoded voxel data.
  const MAX_TOTAL_VOXELS = 64 * 1024 * 1024;

  const root = document.getElementById("rhuh-browser-viewer") ||
    document.getElementById("rhuh-viewer") || document.createElement("main");
  if (!root.parentNode) {
    root.id = "rhuh-browser-viewer";
    (document.body || document.documentElement).appendChild(root);
  }
  root.classList.add("rhuh-browser-viewer");
  root.replaceChildren();

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  const style = element("style");
  style.textContent = `
    .rhuh-browser-viewer { color:#e8edf5; background:#101722; padding:20px;
      border:1px solid #344256; border-radius:14px; font:15px/1.45 system-ui,sans-serif; }
    .rhuh-browser-viewer * { box-sizing:border-box; }
    .rhuh-browser-viewer h2 { margin:0 0 8px; font-size:22px; }
    .rhuh-browser-viewer .viewer-toolbar { display:flex; gap:20px;
      align-items:center; flex-wrap:wrap; margin:16px 0; }
    .rhuh-browser-viewer label { display:flex; gap:10px; align-items:center; }
    .rhuh-browser-viewer select { padding:7px; color:#edf2fa; background:#182334;
      border:1px solid #41536d; max-width:100%; }
    .rhuh-browser-viewer option { color:#edf2fa; background:#182334;
      border-color:#41536d; }
    .rhuh-browser-viewer input[type=range] { accent-color:#78b8ff; min-width:110px; }
    .rhuh-browser-viewer input:disabled,.rhuh-browser-viewer select:disabled { opacity:.5; }
    .rhuh-browser-viewer .viewer-grid { display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; align-items:start; }
    .rhuh-browser-viewer .plane { min-width:0; background:#070b11;
      padding:12px; border:1px solid #344256; border-radius:9px; }
    .rhuh-browser-viewer .plane h3 { font-size:16px; margin:0 0 10px; }
    .rhuh-browser-viewer .plane-controls { display:flex; flex-wrap:wrap;
      justify-content:space-between; gap:6px; margin-bottom:10px; }
    .rhuh-browser-viewer .plane-controls input { width:100%; }
    .rhuh-browser-viewer canvas { display:block; width:100%; height:auto;
      background:black; image-rendering:pixelated; touch-action:pan-y; }
    .rhuh-browser-viewer .axis-caption { color:#aebdd1; font-size:12px; margin:6px 0 0; }
    .rhuh-browser-viewer .viewer-status { padding:10px 12px; margin:12px 0;
      border-radius:7px; background:#1b2a3d; overflow-wrap:anywhere; }
    .rhuh-browser-viewer .viewer-status[data-error=true] { background:#482028; color:#ffd5da; }
    .rhuh-browser-viewer .viewer-notice { color:#c0cede; font-size:13px; margin:8px 0; }
    .rhuh-browser-viewer .viewer-legend { display:flex; flex-wrap:wrap;
      gap:8px 18px; margin:14px 0 6px; }
    .rhuh-browser-viewer .legend-entry { display:inline-flex; align-items:center; gap:7px; }
    .rhuh-browser-viewer .legend-swatch { width:15px; height:15px;
      border:1px solid #93a6bd; border-radius:3px; display:inline-block; }
    @media(max-width:800px) {
      .rhuh-browser-viewer .viewer-grid { grid-template-columns:1fr; }
      .rhuh-browser-viewer { padding:12px; }
      .rhuh-browser-viewer .plane { max-width:560px; width:100%; margin:auto; }
    }
  `;
  root.appendChild(style);

  const status = element("p", "Đang đọc dữ liệu…", "viewer-status");
  status.id = "viewer-status";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  root.appendChild(status);

  let phaseSelect = null;
  let opacity = null;
  let panels = [];
  let active = null;
  let requestSequence = 0;
  let totalRenderCount = 0;

  function setStatus(message, error) {
    status.textContent = message;
    status.dataset.error = error ? "true" : "false";
  }

  function clearCanvases() {
    for (const panel of panels) {
      const context = panel.canvas.getContext("2d");
      if (context) context.clearRect(0, 0, panel.canvas.width, panel.canvas.height);
      panel.canvas.dataset.slice = "";
      panel.canvas.dataset.timepoint = "";
      panel.heading.textContent = panel.name;
    }
  }

  function disableControls(disabled) {
    if (phaseSelect) phaseSelect.disabled = disabled;
    if (opacity) opacity.disabled = disabled;
    for (const panel of panels) panel.slider.disabled = disabled;
  }

  function showError(error) {
    active = null;
    clearCanvases();
    disableControls(true);
    // A bad case must not prevent choosing another preloaded timepoint.
    if (phaseSelect) phaseSelect.disabled = false;
    root.dataset.state = "error";
    root.setAttribute("aria-busy", "false");
    setStatus("Không thể hiển thị: " + (error && error.message ? error.message : String(error)), true);
  }

  function validateCase(item) {
    if (!item || !Array.isArray(item.shape) || item.shape.length !== 3 ||
        !item.shape.every(n => Number.isInteger(n) && n > 0)) {
      throw new Error("Shape phải gồm ba kích thước nguyên dương.");
    }
    const length = item.shape.reduce((product, n) => product * n, 1);
    if (!Number.isSafeInteger(length) || length > MAX_TOTAL_VOXELS) {
      throw new Error("Volume vượt giới hạn kích thước của bộ xem.");
    }
    if (!Array.isArray(item.spacing) || item.spacing.length !== 3 ||
        !item.spacing.every(n => Number.isFinite(n) && n > 0)) {
      throw new Error("Spacing phải gồm ba giá trị dương.");
    }
    if (!Array.isArray(item.initial_indices) || item.initial_indices.length !== 3 ||
        !item.initial_indices.every((n, axis) => Number.isInteger(n) && n >= 0 && n < item.shape[axis])) {
      throw new Error("Vị trí lát ban đầu nằm ngoài volume.");
    }
    for (const field of ["image", "mask"]) {
      const buffer = item[field];
      if (!buffer || buffer.encoding !== "gzip-base64" ||
          typeof buffer.data !== "string" || buffer.byte_length !== length) {
        throw new Error("Buffer " + field + " không đúng định dạng hoặc kích thước.");
      }
    }
    if (!Array.isArray(item.palette)) throw new Error("Thiếu bảng màu nhãn.");
    const colors = new Uint8Array(256 * 3);
    const known = new Uint8Array(256);
    const ids = new Map([[0, 0]]);
    const entries = [];
    known[0] = 1;
    for (const entry of item.palette) {
      if (!entry || !Number.isInteger(entry.index) || entry.index < 0 || entry.index > 255 ||
          !Number.isSafeInteger(entry.id) || !Array.isArray(entry.color) ||
          entry.color.length !== 3 || !entry.color.every(Number.isFinite)) {
        throw new Error("Bảng màu nhãn không hợp lệ.");
      }
      if (entry.index === 0) continue; // Background is always transparent.
      if (known[entry.index]) throw new Error("Bảng màu có index bị trùng.");
      const normalized = entry.color.every(n => n >= 0 && n <= 1);
      const rgb = entry.color.map(n => Math.round(Math.max(0, Math.min(255, n * (normalized ? 255 : 1)))));
      colors.set(rgb, entry.index * 3);
      known[entry.index] = 1;
      ids.set(entry.index, entry.id);
      entries.push({index: entry.index, id: entry.id, color: rgb, name: entry.name});
    }
    return {item, length, colors, known, ids, entries};
  }

  async function decode(buffer) {
    if (typeof DecompressionStream !== "function") {
      throw new Error("Trình duyệt chưa hỗ trợ giải nén gzip. Hãy mở file bằng Chrome hoặc Edge mới.");
    }
    const binary = atob(buffer.data);
    const compressed = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) compressed[i] = binary.charCodeAt(i);
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    if (bytes.byteLength !== buffer.byte_length) {
      throw new Error("Số byte sau giải nén không khớp dữ liệu đã khai báo.");
    }
    return bytes;
  }

  function voxelAt(panel, column, row, indices, shape) {
    if (panel.axis === 0) return [indices[0], column, shape[2] - 1 - row];
    if (panel.axis === 1) return [column, indices[1], shape[2] - 1 - row];
    return [column, shape[1] - 1 - row, indices[2]];
  }

  function render() {
    if (!active) return;
    try {
      const {item, image, mask, colors} = active;
      const indices = panels.map(panel => Number(panel.slider.value));
      const alpha = Number(opacity.value);
      const [, ny, nz] = item.shape;
      for (const panel of panels) {
        const width = item.shape[panel.dims[0]];
        const height = item.shape[panel.dims[1]];
        const context = panel.canvas.getContext("2d");
        if (!context) throw new Error("Không khởi tạo được Canvas 2D.");
        const pixels = context.createImageData(width, height);
        for (let row = 0; row < height; row++) {
          for (let column = 0; column < width; column++) {
            // NumPy C order: X,Y,Z. Avoid allocating a coordinate array per pixel.
            const offset = panel.axis === 0 ?
              (indices[0] * ny + column) * nz + (nz - 1 - row) :
              panel.axis === 1 ?
                (column * ny + indices[1]) * nz + (nz - 1 - row) :
                (column * ny + (ny - 1 - row)) * nz + indices[2];
            const grey = image[offset];
            const label = mask[offset];
            const pixel = (row * width + column) * 4;
            // putImageData ignores globalAlpha, so blend every RGB pixel here.
            for (let channel = 0; channel < 3; channel++) {
              pixels.data[pixel + channel] = label === 0 ? grey :
                Math.round(grey * (1 - alpha) + colors[label * 3 + channel] * alpha);
            }
            pixels.data[pixel + 3] = 255;
          }
        }
        context.putImageData(pixels, 0, 0);
        panel.heading.textContent = panel.name + " · index=" + indices[panel.axis];
        panel.value.textContent = indices[panel.axis] + " / " + (item.shape[panel.axis] - 1);
        panel.canvas.dataset.slice = String(indices[panel.axis]);
        panel.canvas.dataset.timepoint = String(item.timepoint);
        panel.canvas.dataset.renderCount = String(Number(panel.canvas.dataset.renderCount || 0) + 1);
      }
      opacityValue.textContent = alpha.toFixed(2);
      totalRenderCount++;
      root.dataset.renderCount = String(totalRenderCount);
      root.dataset.timepoint = String(item.timepoint);
      setStatus(payload.patient + " / " + item.timepoint + " / " + payload.modality +
        " | x=" + indices[0] + ", y=" + indices[1] + ", z=" + indices[2] +
        " | opacity=" + alpha.toFixed(2), false);
    } catch (error) {
      showError(error);
    }
  }

  let payload;
  let opacityValue;
  let notice;
  let legend;
  let hover;
  const decoded = new Map();

  async function loadCase(index) {
    const sequence = ++requestSequence;
    active = null;
    disableControls(true);
    clearCanvases();
    legend.replaceChildren();
    notice.textContent = "";
    hover.textContent = "";
    root.dataset.state = "loading";
    root.dataset.timepoint = "";
    root.setAttribute("aria-busy", "true");
    setStatus("Đang giải nén…", false);
    try {
      if (!Number.isInteger(index) || index < 0 || index >= payload.cases.length) {
        throw new Error("Timepoint đã chọn không có trong dữ liệu.");
      }
      setStatus("Đang giải nén " + String(payload.cases[index].title || payload.cases[index].timepoint) + "…", false);
      let volume = decoded.get(index);
      if (!volume) {
        const metadata = validateCase(payload.cases[index]);
        const [image, mask] = await Promise.all([
          decode(metadata.item.image), decode(metadata.item.mask),
        ]);
        for (let i = 0; i < mask.length; i++) {
          if (!metadata.known[mask[i]]) throw new Error("Mask có index không có trong bảng màu: " + mask[i]);
        }
        volume = {...metadata, image, mask};
        decoded.set(index, volume);
      }
      if (sequence !== requestSequence) return;
      active = volume;
      for (const panel of panels) {
        const {item} = volume;
        panel.slider.max = String(item.shape[panel.axis] - 1);
        panel.slider.value = String(item.initial_indices[panel.axis]);
        panel.canvas.width = item.shape[panel.dims[0]];
        panel.canvas.height = item.shape[panel.dims[1]];
        panel.canvas.style.aspectRatio =
          (item.shape[panel.dims[0]] * item.spacing[panel.dims[0]]) + " / " +
          (item.shape[panel.dims[1]] * item.spacing[panel.dims[1]]);
      }
      for (const entry of volume.entries) {
        const chip = element("span", undefined, "legend-entry");
        const swatch = element("span", undefined, "legend-swatch");
        swatch.style.backgroundColor = "rgb(" + entry.color.join(",") + ")";
        swatch.setAttribute("aria-hidden", "true");
        const id = "ID " + entry.id;
        const name = typeof entry.name === "string" && entry.name ? entry.name : id;
        chip.append(swatch, element("span", name === id ? id : name + " (" + id + ")"));
        legend.appendChild(chip);
      }
      const notices = [];
      if (volume.item.unit_notice) notices.push(String(volume.item.unit_notice));
      if (volume.item.resampled) notices.push("Ảnh và nhãn đã được resample riêng cho hiển thị RAS.");
      notice.textContent = notices.join(" ");
      disableControls(false);
      root.dataset.state = "ready";
      root.setAttribute("aria-busy", "false");
      render();
    } catch (error) {
      if (sequence === requestSequence) showError(error);
    }
  }

  try {
    const source = document.getElementById("rhuh-viewer-data");
    if (!source) throw new Error("Không tìm thấy dữ liệu rhuh-viewer-data.");
    payload = JSON.parse(source.textContent);
    if (!payload || payload.schema !== 1 || !Array.isArray(payload.cases) || !payload.cases.length) {
      throw new Error("Dữ liệu không đúng schema 1 hoặc không có timepoint.");
    }
    if (typeof payload.patient !== "string" || typeof payload.modality !== "string") {
      throw new Error("Thiếu patient hoặc modality.");
    }
    let totalVoxels = 0;
    for (const item of payload.cases) {
      totalVoxels += validateCase(item).length;
      if (totalVoxels > MAX_TOTAL_VOXELS) {
        throw new Error("Tổng volume vượt giới hạn 64 Mi voxel của bộ xem.");
      }
    }
    const timepoints = payload.cases.map(item => String(item && item.timepoint));
    if (new Set(timepoints).size !== timepoints.length) throw new Error("Timepoint bị trùng.");

    const heading = element("h2", payload.patient + " · " + payload.modality);
    root.insertBefore(heading, status);
    const toolbar = element("div", undefined, "viewer-toolbar");
    const phaseLabel = element("label", "Thời điểm");
    phaseSelect = element("select");
    phaseSelect.id = "timepoint";
    phaseSelect.setAttribute("aria-label", "Thời điểm");
    payload.cases.forEach((item, index) => {
      const option = element("option", String(item.title || "Thời điểm " + item.timepoint));
      option.value = String(item.timepoint);
      option.dataset.timepoint = String(item.timepoint);
      phaseSelect.appendChild(option);
    });
    phaseLabel.appendChild(phaseSelect);
    const opacityLabel = element("label", "Độ đậm nhãn");
    opacity = element("input");
    opacity.type = "range";
    opacity.id = "opacity";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.05";
    opacity.value = "0.4";
    opacity.setAttribute("aria-label", "Độ đậm nhãn");
    opacityValue = element("output", "0.40");
    opacityValue.setAttribute("for", "opacity");
    opacityLabel.append(opacity, opacityValue);
    toolbar.append(phaseLabel, opacityLabel);
    root.insertBefore(toolbar, status);

    const grid = element("div", undefined, "viewer-grid");
    const specifications = [
      {key: "sagittal", name: "Sagittal", axis: 0, dims: [1, 2], caption: "Ngang: P → A · Dọc: I → S"},
      {key: "coronal", name: "Coronal", axis: 1, dims: [0, 2], caption: "Ngang: L → R · Dọc: I → S"},
      {key: "axial", name: "Axial", axis: 2, dims: [0, 1], caption: "Ngang: L → R · Dọc: P → A"},
    ];
    panels = specifications.map(specification => {
      const panel = {...specification};
      const section = element("section", undefined, "plane");
      panel.heading = element("h3", panel.name);
      const controls = element("label", undefined, "plane-controls");
      controls.appendChild(element("span", "Lát " + panel.name));
      panel.value = element("output", "—");
      panel.value.setAttribute("for", panel.key);
      panel.slider = element("input");
      panel.slider.type = "range";
      panel.slider.id = panel.key;
      panel.slider.min = "0";
      panel.slider.max = "0";
      panel.slider.step = "1";
      panel.slider.value = "0";
      panel.slider.setAttribute("aria-label", "Lát " + panel.name);
      controls.append(panel.value, panel.slider);
      panel.canvas = element("canvas");
      panel.canvas.id = "canvas-" + panel.key;
      panel.canvas.width = 1;
      panel.canvas.height = 1;
      panel.canvas.textContent = "Ảnh " + panel.name + " cần trình duyệt hỗ trợ Canvas.";
      panel.canvas.setAttribute("aria-label", "Ảnh " + panel.name + " với nhãn");
      panel.canvas.dataset.renderCount = "0";
      section.append(panel.heading, controls, panel.canvas, element("p", panel.caption, "axis-caption"));
      grid.appendChild(section);
      panel.slider.addEventListener("input", render);
      panel.canvas.addEventListener("pointermove", event => {
        if (!active) return;
        const box = panel.canvas.getBoundingClientRect();
        if (!box.width || !box.height) return;
        const column = Math.max(0, Math.min(panel.canvas.width - 1,
          Math.floor((event.clientX - box.left) * panel.canvas.width / box.width)));
        const row = Math.max(0, Math.min(panel.canvas.height - 1,
          Math.floor((event.clientY - box.top) * panel.canvas.height / box.height)));
        const indices = panels.map(other => Number(other.slider.value));
        const [x, y, z] = voxelAt(panel, column, row, indices, active.item.shape);
        const offset = (x * active.item.shape[1] + y) * active.item.shape[2] + z;
        hover.textContent = panel.name + " | x=" + x + ", y=" + y + ", z=" + z +
          " | Nhãn: ID " + active.ids.get(active.mask[offset]);
      });
      return panel;
    });
    root.appendChild(grid);
    legend = element("div", undefined, "viewer-legend");
    legend.setAttribute("aria-label", "Bảng màu nhãn");
    notice = element("p", "", "viewer-notice");
    hover = element("p", "", "viewer-notice");
    hover.id = "viewer-hover";
    root.append(legend, notice, hover);
    opacity.addEventListener("input", render);
    phaseSelect.addEventListener("change", () => loadCase(payload.cases.findIndex(
      item => String(item.timepoint) === phaseSelect.value
    )));
    loadCase(0);
  } catch (error) {
    showError(error);
  }
})();
