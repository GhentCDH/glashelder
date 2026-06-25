import http.server
import json
import socketserver
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── Easily adaptable: add / remove types here ─────────────────────────────────
ITEM_TYPES: list[str] = ["text", "creator", "title", "place", "institute", "date"]

# Static SPA — no Python format placeholders; JS braces don't need escaping.
_SPA_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>annotate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; border: 0 solid; }

  body {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    background: #fff;
    color: #000;
    height: 100dvh;
    overflow: hidden;
  }

  /* ── full-page loading screen ── */
  #global-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100dvh;
    gap: 20px;
    color: #888;
    font-size: 12px;
    letter-spacing: 0.1em;
  }
  #global-loading.hidden { display: none; }

  /* ── annotation layout ── */
  #ann-view { display: none; height: 100dvh; }
  #ann-view.visible { display: flex; }

  /* ── image pane ── */
  #img-pane {
    flex: 1;
    position: relative;
    background: #111;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  #the-img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  #the-img.loaded { opacity: 1; }
  #img-loading {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    color: #888;
    font-size: 12px;
    letter-spacing: 0.1em;
    pointer-events: none;
  }
  #img-loading.hidden { display: none; }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    width: 32px;
    height: 32px;
    border: 2px solid #333;
    border-top-color: #aaa;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  /* ── annotation pane ── */
  #ann-pane {
    width: 460px;
    min-width: 420px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 20px;
    overflow-y: auto;
    border-left: 1px solid #000;
  }

  .progress { font-size: 11px; color: #555; word-break: break-all; }

  /* ── list sections ── */
  .section { display: flex; flex-direction: column; gap: 6px; flex: 1; }
  .section-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    border-bottom: 1px solid #000;
    padding-bottom: 4px;
  }

  .item-list {
    list-style: none;
    min-height: 48px;
    max-height: 150px;
    overflow-y: auto;
    border: 1px solid #ccc;
  }
  .item-list:empty::after {
    content: '— empty —';
    display: block;
    padding: 6px 8px;
    color: #bbb;
    font-style: italic;
    font-size: 12px;
  }
  .item-list li {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px 3px 4px;
    border-bottom: 1px solid #f0f0f0;
    line-height: 1.5;
    cursor: grab;
  }
  .item-list li:last-child { border-bottom: none; }
  .item-list li.dragging { opacity: 0.35; }
  .item-list.drag-over { background: #f5f5f5; outline: 2px dashed #888; outline-offset: -2px; }

  .drag-handle {
    flex-shrink: 0;
    color: #ccc;
    font-size: 12px;
    user-select: none;
    padding: 0 2px;
  }

  .item-text {
    flex: 1;
    cursor: pointer;
    min-width: 0;
    word-break: break-word;
  }
  .item-text:hover { text-decoration: underline dotted #aaa; }

  .item-type {
    flex-shrink: 0;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #777;
    border: 1px solid #ddd;
    padding: 1px 5px;
    white-space: nowrap;
  }

  .del-btn {
    flex-shrink: 0;
    background: none;
    border: none;
    color: #ccc;
    font-family: inherit;
    font-size: 14px;
    line-height: 1;
    cursor: pointer;
    padding: 0 2px;
  }
  .del-btn:hover { color: #c00; }

  /* ── input area ── */
  #input-area {
    display: flex;
    flex-direction: column;
    gap: 6px;
    border: 1px solid #000;
    padding: 10px;
  }
  #input-row { display: flex; }
  #main-input {
    flex: 1;
    padding: 7px 8px;
    border: 1px solid #ccc;
    font-family: inherit;
    font-size: 13px;
    outline: none;
  }
  #main-input:focus { background: #fffde7; border-color: #000; }

  #type-select {
    padding: 7px 6px;
    border: 1px solid #ccc;
    border-left: none;
    font-family: inherit;
    font-size: 12px;
    background: #f8f8f8;
    cursor: pointer;
    outline: none;
    color: #444;
  }
  #type-select:focus { border-color: #000; background: #fff; }

  #add-btns { display: flex; }
  #add-btns button {
    flex: 1;
    padding: 7px 6px;
    border: 1px solid #000;
    border-right: none;
    background: #fff;
    color: #000;
    font-family: inherit;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  #add-btns button:last-child { border-right: 1px solid #000; }
  #add-btns button:hover { background: #000; color: #fff; }

  /* ── action buttons ── */
  #btn-row { display: flex; border: 1px solid #000; }
  #btn-row button {
    flex: 1;
    padding: 10px 4px;
    border: none;
    border-right: 1px solid #000;
    background: #fff;
    color: #000;
    font-family: inherit;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  #btn-row button:last-child { border-right: none; }
  #btn-row button:hover { background: #000; color: #fff; }
  #btn-save:hover { background: #1a5c1a !important; color: #fff !important; }
  #btn-quit:hover { background: #7a1a1a !important; color: #fff !important; }

  #hint { font-size: 11px; color: #999; line-height: 1.8; }
  kbd { border: 1px solid #bbb; padding: 1px 4px; font-family: inherit; font-size: 10px; }
</style>
</head>
<body>

<div id="global-loading">
  <div class="spinner"></div>
  <span id="global-loading-msg">connecting&hellip;</span>
</div>

<div id="ann-view">
  <div id="img-pane">
    <div id="img-loading">
      <div class="spinner"></div>
      <span>loading image&hellip;</span>
    </div>
    <img id="the-img" src="" alt="" onload="onImgLoad()" onerror="onImgError()">
  </div>

  <div id="ann-pane">
    <div class="progress" id="progress"></div>

    <div class="section">
      <div class="section-label">Content &mdash; body text</div>
      <ul id="content-list" class="item-list"
          ondragover="onDragOver('content', event)"
          ondragleave="onDragLeave('content', event)"
          ondrop="onDrop('content', event)"></ul>
    </div>

    <div class="section">
      <div class="section-label">Meta &mdash; margin / label text</div>
      <ul id="meta-list" class="item-list"
          ondragover="onDragOver('meta', event)"
          ondragleave="onDragLeave('meta', event)"
          ondrop="onDrop('meta', event)"></ul>
    </div>

    <div id="input-area">
      <div id="input-row">
        <input id="main-input" type="text" placeholder="type a line&hellip;" autocomplete="off">
        <select id="type-select"></select>
      </div>
      <div id="add-btns">
        <button onclick="addItem('content')">+ Content</button>
        <button onclick="addItem('meta')">+ Meta</button>
      </div>
    </div>

    <div id="btn-row">
      <button id="btn-undo" onclick="undo()">Undo</button>
      <button id="btn-skip" onclick="submit('skip')">Skip</button>
      <button id="btn-save" onclick="submit('save')">Save</button>
      <button id="btn-quit" onclick="submit('quit')">Quit</button>
    </div>

    <div id="hint">
      <kbd>Ctrl 1</kbd> add to content &nbsp;&middot;&nbsp;
      <kbd>Ctrl 2</kbd> add to meta &nbsp;&middot;&nbsp;
      <kbd>Ctrl Z</kbd> undo<br>
      <kbd>Ctrl S</kbd> save &nbsp;&middot;&nbsp;
      <kbd>Esc</kbd> skip &nbsp;&middot;&nbsp;
      click item to re-edit
    </div>
  </div>
</div>

<script>
  // items are {text: string, type: string}
  let contentItems = [];
  let metaItems    = [];
  const undoStack  = [];

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function listFor(field) {
    return field === 'content' ? contentItems : metaItems;
  }

  function renderList(field) {
    document.getElementById(field + '-list').innerHTML =
      listFor(field).map((e, i) => `
        <li draggable="true"
            ondragstart="onDragStart('${field}', ${i}, event)"
            ondragend="onDragEnd(event)">
          <span class="drag-handle">&#8942;&#8942;</span>
          <span class="item-text" onclick="copyToInput('${field}', ${i})">${esc(e.text)}</span>
          <span class="item-type">${esc(e.type)}</span>
          <button class="del-btn" onclick="deleteItem('${field}', ${i})" title="delete">&times;</button>
        </li>`).join('');
  }

  function render() {
    renderList('content');
    renderList('meta');
  }

  function currentType() {
    return document.getElementById('type-select').value;
  }

  function addItem(field) {
    const input = document.getElementById('main-input');
    const val = input.value.trim();
    if (!val) return;
    const entry = { text: val, type: currentType() };
    const items = listFor(field);
    const index = items.length;
    items.push(entry);
    undoStack.push({ op: 'add', field, index, item: entry });
    input.value = '';
    renderList(field);
    input.focus();
  }

  function deleteItem(field, index) {
    const items = listFor(field);
    const item = items.splice(index, 1)[0];
    undoStack.push({ op: 'delete', field, index, item });
    renderList(field);
  }

  function copyToInput(field, index) {
    const items = listFor(field);
    const item = items.splice(index, 1)[0];
    undoStack.push({ op: 'delete', field, index, item });
    document.getElementById('main-input').value = item.text;
    document.getElementById('type-select').value = item.type;
    renderList(field);
    const input = document.getElementById('main-input');
    input.focus();
    input.select();
  }

  function undo() {
    if (!undoStack.length) return;
    const entry = undoStack.pop();
    if (entry.op === 'add') {
      listFor(entry.field).splice(entry.index, 1);
      renderList(entry.field);
    } else if (entry.op === 'delete') {
      listFor(entry.field).splice(entry.index, 0, entry.item);
      renderList(entry.field);
    } else if (entry.op === 'move') {
      listFor(entry.toField).splice(entry.toIndex, 1);
      listFor(entry.fromField).splice(entry.fromIndex, 0, entry.item);
      render();
    }
  }

  // ── drag and drop ──────────────────────────────────────────────────────────
  let dragSrc = null; // {field, index, item}

  function onDragStart(field, index, e) {
    dragSrc = { field, index, item: listFor(field)[index] };
    e.dataTransfer.effectAllowed = 'move';
    // Defer adding the class so the browser snapshot is taken first
    setTimeout(() => e.target.closest('li').classList.add('dragging'), 0);
  }

  function onDragEnd(e) {
    e.target.closest('li').classList.remove('dragging');
    document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
    dragSrc = null;
  }

  function onDragOver(field, e) {
    if (!dragSrc || dragSrc.field === field) return; // only cross-list
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    document.getElementById(field + '-list').classList.add('drag-over');
  }

  function onDragLeave(field, e) {
    // Only remove when leaving the list itself, not a child element
    if (!e.currentTarget.contains(e.relatedTarget)) {
      document.getElementById(field + '-list').classList.remove('drag-over');
    }
  }

  function onDrop(targetField, e) {
    e.preventDefault();
    if (!dragSrc || dragSrc.field === targetField) return;
    document.getElementById(targetField + '-list').classList.remove('drag-over');

    const srcItems = listFor(dragSrc.field);
    srcItems.splice(dragSrc.index, 1);
    const dstItems = listFor(targetField);
    const toIndex = dstItems.length;
    dstItems.push(dragSrc.item);

    undoStack.push({ op: 'move', fromField: dragSrc.field, fromIndex: dragSrc.index, toField: targetField, toIndex, item: dragSrc.item });
    render();
  }

  function onImgLoad() {
    document.getElementById('the-img').classList.add('loaded');
    document.getElementById('img-loading').classList.add('hidden');
  }

  function onImgError() {
    document.getElementById('img-loading').innerHTML =
      '<span style="color:#c00">failed to load image</span>';
  }

  function showGlobalLoading(msg) {
    document.getElementById('global-loading-msg').textContent = msg;
    document.getElementById('global-loading').classList.remove('hidden');
    document.getElementById('ann-view').classList.remove('visible');
  }

  function showAnnotationView() {
    document.getElementById('global-loading').classList.add('hidden');
    document.getElementById('ann-view').classList.add('visible');
  }

  async function fetchState() {
    try {
      const res = await fetch('/state');
      if (res.status === 503) return fetchState();
      if (!res.ok) { setTimeout(fetchState, 2000); return; }
      const state = await res.json();

      // Populate type dropdown from server-defined list
      const sel = document.getElementById('type-select');
      sel.innerHTML = state.item_types
        .map(t => `<option value="${esc(t)}">${esc(t)}</option>`)
        .join('');

      // Reset per-image state
      contentItems = state.seed_content.map(e => ({ ...e }));
      metaItems    = state.seed_meta.map(e => ({ ...e }));
      undoStack.length = 0;
      document.getElementById('main-input').value = '';

      document.getElementById('progress').textContent =
        `${state.name} · ${state.i} / ${state.count}`;
      document.title = `annotate :: ${state.name} (${state.i}/${state.count})`;

      const img = document.getElementById('the-img');
      img.classList.remove('loaded');
      const imgLoading = document.getElementById('img-loading');
      imgLoading.classList.remove('hidden');
      imgLoading.innerHTML = '<div class="spinner"></div><span>loading image…</span>';
      img.src = '/image?t=' + Date.now();

      render();
      showAnnotationView();
      document.getElementById('main-input').focus();
    } catch (e) {
      setTimeout(fetchState, 2000);
    }
  }

  async function submit(action) {
    showGlobalLoading(
      action === 'save' ? 'saving…' : action === 'quit' ? 'quitting…' : 'skipping…'
    );
    try {
      await fetch('/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, content: contentItems, meta: metaItems }),
      });
    } catch (e) { /* server may close connection */ }

    if (action === 'quit') {
      document.getElementById('global-loading').querySelector('.spinner').style.display = 'none';
      document.getElementById('global-loading-msg').textContent =
        'session complete — you can close this tab.';
      return;
    }
    fetchState();
  }

  document.addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === '1') { e.preventDefault(); addItem('content'); }
    if (e.ctrlKey && e.key === '2') { e.preventDefault(); addItem('meta'); }
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); submit('save'); }
    if (e.key === 'Escape') submit('skip');
  });

  fetchState();
</script>
</body>
</html>
"""

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class AnnotationEntry:
    text: str
    type: str = "text"


@dataclass
class Annotation:
    meta: list[AnnotationEntry] = field(default_factory=list)
    content: list[AnnotationEntry] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.meta and not self.content

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Annotation":
        d = json.loads(path.read_text())
        return cls(
            meta=[AnnotationEntry(**e) for e in d.get("meta", [])],
            content=[AnnotationEntry(**e) for e in d.get("content", [])],
        )


# section headers in the editor buffer
_CONTENT = "[content]"
_META = "[meta]"
QUIT = "quit"


def annotation_from_ocr(text: str) -> Annotation:
    """VLM OUTPUT_SCHEMA json -> Annotation. Caller wraps in try/except."""
    d = json.loads(text)
    return Annotation(
        meta=[
            AnnotationEntry(text=e["text"], type=e.get("type", "text"))
            for e in d.get("meta", [])
        ],
        content=[
            AnnotationEntry(text=e["text"], type=e.get("type", "text"))
            for e in d.get("content", [])
        ],
    )


def render_buffer(name: str, seed: Annotation, i: int, count: int) -> str:
    """Editor buffer pre-filled with the VLM transcription for the user to correct."""
    lines = [
        f"# image: {name}  ({i}/{count}) - pre-filled by VLM, correct as needed",
        "# Lines under [content]/[meta]: 'text|type' (type defaults to 'text').",
        f"# Save & close to store. Empty = skip. A line '{QUIT}' or :cq = stop.",
        _CONTENT,
        *[f"{e.text}|{e.type}" for e in seed.content],
        _META,
        *[f"{e.text}|{e.type}" for e in seed.meta],
    ]
    return "\n".join(lines) + "\n"


def parse_buffer(text: str) -> tuple[Annotation, bool]:
    """Parse edited buffer -> (Annotation, quit). Format per line: 'text|type'."""
    ann = Annotation()
    bucket = ann.content
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == QUIT:
            return ann, True
        if line == _CONTENT:
            bucket = ann.content
            continue
        if line == _META:
            bucket = ann.meta
            continue
        parts = raw.rstrip().rsplit("|", 1)
        bucket.append(
            AnnotationEntry(text=parts[0], type=parts[1] if len(parts) > 1 else "text")
        )
    return ann, False


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCPServer that handles each request in its own thread, enabling long-polling."""

    daemon_threads = True
    allow_reuse_address = True


class _Session:
    """
    Persistent annotation server. Lives for the whole CLI session so the
    browser tab stays open and just refreshes its content between images.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_image: Path | None = None
        self._state: dict | None = None
        self._state_ready = threading.Event()
        self._result_ready = threading.Event()
        self._result: dict = {}

        self._server = _ThreadingServer(("127.0.0.1", 0), self._make_handler())
        self._port: int = self._server.server_address[1]
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/"

    def set_image(self, image_path: Path, seed: Annotation, i: int, count: int) -> None:
        self._result_ready.clear()
        self._current_image = image_path
        with self._lock:
            self._state = {
                "name": image_path.name,
                "i": i,
                "count": count,
                "item_types": ITEM_TYPES,
                "seed_content": [asdict(e) for e in seed.content],
                "seed_meta": [asdict(e) for e in seed.meta],
            }
        self._state_ready.set()

    def wait_for_result(self) -> dict:
        self._result_ready.wait()
        return self._result.copy()

    def _make_handler(self) -> type:
        session = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0]

                if path == "/":
                    body = _SPA_HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif path == "/state":
                    got = session._state_ready.wait(timeout=300)
                    if not got or session._state is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    with session._lock:
                        state = session._state.copy()
                    session._state_ready.clear()
                    body = json.dumps(state).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif path == "/image":
                    img = session._current_image
                    if img is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    data = img.read_bytes()
                    mime = _MIME.get(img.suffix.lower(), "application/octet-stream")
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                if self.path == "/action":
                    length = int(self.headers.get("Content-Length", 0))
                    session._result = json.loads(self.rfile.read(length))
                    self.send_response(200)
                    self.end_headers()
                    session._result_ready.set()
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        return _Handler


_session: _Session | None = None


def annotate_with_gui(
    image_path: Path,
    seed: Annotation,
    i: int,
    count: int,
) -> tuple[Annotation | None, bool]:
    """
    Open a localhost annotation UI (once) and block until the user acts.

    Returns (annotation, quit_flag):
      (Annotation, False)  - saved, continue
      (None,       False)  - skipped, continue
      (None,       True)   - quit
    """
    import webbrowser

    global _session
    if _session is None:
        _session = _Session()
        print(f"  annotate -> {_session.url}")
        webbrowser.open(_session.url)

    _session.set_image(image_path, seed, i, count)
    result = _session.wait_for_result()

    action = result.get("action")
    if action == "quit":
        return None, True
    if action == "skip":
        return None, False
    return Annotation(
        content=[
            AnnotationEntry(text=e["text"], type=e["type"]) for e in result["content"]
        ],
        meta=[AnnotationEntry(text=e["text"], type=e["type"]) for e in result["meta"]],
    ), False


def _selfcheck() -> None:
    a = annotation_from_ocr(
        '{"meta":[{"text":"m1"}],"content":[{"text":"c1"},{"text":"c2"}]}'
    )
    assert a == Annotation(
        meta=[AnnotationEntry("m1")],
        content=[AnnotationEntry("c1"), AnnotationEntry("c2")],
    ), a

    buf = "# comment\n[content]\nhello|creator\nworld\n[meta]\nmargin|place\n"
    ann, quit = parse_buffer(buf)
    assert not quit
    assert ann == Annotation(
        meta=[AnnotationEntry("margin", "place")],
        content=[AnnotationEntry("hello", "creator"), AnnotationEntry("world", "text")],
    ), ann

    ann2, _ = parse_buffer(render_buffer("x.jpg", ann, 0, 5))
    assert ann2 == ann, ann2

    _, q = parse_buffer("[content]\nquit\n")
    assert q

    print("ok")


if __name__ == "__main__":
    _selfcheck()
