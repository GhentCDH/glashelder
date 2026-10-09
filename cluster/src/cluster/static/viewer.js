import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const GOLDEN = 0.618033988749895;
const BACKGROUND = 0x171717;
const NOISE = [110, 110, 110];
const DIM = [55, 55, 55];
const SPRITE_SIZE = 1 / 30; // of the radius of the whole cloud
const FLIGHT_MS = 600;
const FADE_MS = 300;
const FOG_REACH = 1.5; // the fog is full this far behind the cloud's centre, in its radii
const MAX_DISTANCE = 10; // how far the camera can zoom out, in cloud radii
const LIST_SAMPLES = 8;
const PANE_SAMPLES = 24;

// colours are used as they are, so a point matches its swatch in the list
THREE.ColorManagement.enabled = false;

function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)].map((v) => Math.round(v * 255));
}
const colourOf = (id) => hsl((id * GOLDEN) % 1, 0.7, 0.55);
const css = ([r, g, b]) => `rgb(${r} ${g} ${b})`;
const element = (tag, className = "", text = "") => {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
};
const number = (x, digits = 2) => (x === null ? "-" : x.toFixed(digits));

const binary = (url, Type) =>
  fetch(url).then((r) => r.arrayBuffer()).then((b) => new Type(b));
const [positions, labels, info] = await Promise.all([
  binary("points", Float32Array),
  binary("labels", Int32Array),
  fetch("clusters").then((r) => r.json()),
]);
const n = labels.length;
const clusters = new Map(info.clusters.map((c) => [c.id, c]));
const colours = new Map(info.clusters.map((c) => [c.id, colourOf(c.id)]));
const groups = info.groups; // name -> cluster ids

// soft clustering: the strongest cluster of each point, and its score
const [softLabels, softScores] = info.has_soft
  ? await Promise.all([
      binary("soft-labels", Int32Array),
      binary("soft-scores", Float32Array),
    ])
  : [null, null];
const soft = info.soft; // {threshold, clusters: {id: threshold}}, as saved
const SOFT_SCALE = 0.65; // size of points that joined a cluster from noise

// the labels that are shown: noise points join their strongest cluster when
// its score reaches that cluster's threshold; points in a cluster stay put
const assigned = new Int32Array(labels);
const joined = new Uint8Array(n);
const extra = new Map(); // cluster id -> points that joined it
const thresholdOf = (id) => soft.clusters[id] ?? soft.threshold;

function assign() {
  extra.clear();
  for (let i = 0; i < n; i++) {
    assigned[i] = labels[i];
    joined[i] = 0;
    if (labels[i] >= 0 || softLabels === null) continue;
    const strongest = softLabels[i];
    const threshold = strongest >= 0 ? thresholdOf(strongest) : null;
    if (threshold === null || threshold === undefined) continue;
    if (softScores[i] < threshold) continue;
    assigned[i] = strongest;
    joined[i] = 1;
    extra.set(strongest, (extra.get(strongest) ?? 0) + 1);
  }
}
const sizeOf = (id) => (clusters.get(id)?.size ?? 0) + (extra.get(id) ?? 0);
const noiseLeft = () => info.noise - [...extra.values()].reduce((a, b) => a + b, 0);
const countText = (id) => {
  const more = extra.get(id);
  const size = clusters.get(id).size.toLocaleString();
  return more ? `${size} +${more.toLocaleString()}` : size;
};

let focus = null; // null, {cluster: id} or {group: name}
const hidden = new Set(); // cluster ids not drawn, -1 is noise
let showImages = false;
const checked = new Set();

const focusIds = () =>
  focus === null
    ? null
    : "cluster" in focus
      ? new Set([focus.cluster])
      : new Set(groups[focus.group] ?? []);

// scene
const view = document.getElementById("view");
const pane = document.getElementById("pane");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio);
view.prepend(renderer.domElement);
document.getElementById("loading").remove();

const scene = new THREE.Scene();
scene.background = new THREE.Color(BACKGROUND);
// near and far follow the camera, see the animation loop
scene.fog = new THREE.Fog(BACKGROUND, 1, 2);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
// a colour change fades from one to the other
for (const name of ["colorFrom", "colorTo"]) {
  geometry.setAttribute(name, new THREE.BufferAttribute(new Uint8Array(n * 3), 3, true));
}
geometry.setAttribute("scale", new THREE.BufferAttribute(new Float32Array(n).fill(1), 1));
geometry.computeBoundingSphere();
const whole = geometry.boundingSphere.clone();

const pointSize = whole.radius / 150;
// round points with a soft edge, sized in the scene, so far ones are smaller
const pointsMaterial = new THREE.ShaderMaterial({
  uniforms: THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      size: { value: pointSize },
      viewportHeight: { value: 1 },
      fade: { value: 1 },
    },
  ]),
  vertexShader: `
    uniform float size;
    uniform float viewportHeight;
    uniform float fade;
    attribute vec3 colorFrom;
    attribute vec3 colorTo;
    attribute float scale;
    varying vec3 vColor;
    #include <fog_pars_vertex>
    void main() {
      vColor = mix(colorFrom, colorTo, fade);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;
      gl_PointSize = max(2.0, scale * size * projectionMatrix[1][1] * viewportHeight * 0.5 / -mvPosition.z);
      #include <fog_vertex>
    }`,
  fragmentShader: `
    varying vec3 vColor;
    #include <fog_pars_fragment>
    void main() {
      float d = length(gl_PointCoord - 0.5);
      if (d > 0.5) discard;
      // a little darker towards the edge, so points on top of each other stay apart
      float shade = 1.0 - 0.35 * smoothstep(0.15, 0.5, d);
      gl_FragColor = vec4(vColor * shade, 1.0 - smoothstep(0.42, 0.5, d));
      #include <fog_fragment>
    }`,
  fog: true,
  // the soft edge becomes multisample coverage: smooth, and no blending, so
  // the order points are drawn in does not matter
  alphaToCoverage: true,
});
const cloud = new THREE.Points(geometry, pointsMaterial);
scene.add(cloud);
const sprites = new THREE.Group();
scene.add(sprites);

const camera = new THREE.PerspectiveCamera(
  50, 1, whole.radius / 1000, whole.radius * (MAX_DISTANCE + 2),
);
camera.position.copy(whole.center).add(new THREE.Vector3(0, 0, whole.radius * 2.2));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(whole.center);
controls.enableDamping = true;
controls.maxDistance = whole.radius * MAX_DISTANCE;
controls.autoRotateSpeed = 1.5;

function resize() {
  const width = view.clientWidth;
  const height = view.clientHeight;
  renderer.setSize(width, height);
  pointsMaterial.uniforms.viewportHeight.value = height * renderer.getPixelRatio();
  camera.aspect = width / height;
  // with the pane open, centre the view on the part above it
  const covered = pane.classList.contains("hidden") ? 0 : pane.offsetHeight;
  if (covered) camera.setViewOffset(width, height + covered, 0, covered, width, height);
  else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);

// camera flights to a sphere, then a slow turn until the user takes over
let flight = null;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

// how far away a sphere fits in the part of the view that the pane leaves free
function fitDistance(radius) {
  const covered = pane.classList.contains("hidden") ? 0 : pane.offsetHeight;
  const width = view.clientWidth;
  const height = view.clientHeight;
  // with a view offset, the fov spans the full frame: the view plus the pane
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) / (height + covered);
  const half = tan * Math.min(height - covered, width);
  return (radius * Math.sqrt(1 + half * half)) / half;
}

function flyTo(sphere, rotate) {
  const direction = camera.position.clone().sub(controls.target).normalize();
  const distance = fitDistance(Math.max(sphere.radius, pointSize * 20)) * 1.1;
  flight = {
    start: performance.now(),
    from: [camera.position.clone(), controls.target.clone()],
    to: [sphere.center.clone().addScaledVector(direction, distance), sphere.center.clone()],
    rotate,
  };
  controls.autoRotate = false;
}

controls.addEventListener("start", () => {
  flight = null;
  controls.autoRotate = false;
});

renderer.setAnimationLoop((now) => {
  if (flight) {
    const t = Math.min(1, (now - flight.start) / FLIGHT_MS);
    const k = ease(t);
    camera.position.lerpVectors(flight.from[0], flight.to[0], k);
    controls.target.lerpVectors(flight.from[1], flight.to[1], k);
    if (t === 1) {
      controls.autoRotate = flight.rotate;
      flight = null;
    }
  }
  controls.update();

  // clear at the front of the cloud, darker towards its back
  const distance = camera.position.distanceTo(whole.center);
  scene.fog.near = Math.max(0, distance - whole.radius);
  scene.fog.far = distance + whole.radius * FOG_REACH;

  pointsMaterial.uniforms.fade.value = ease(Math.min(1, (now - fadeStart) / FADE_MS));
  for (const sprite of sprites.children) {
    const { material, userData } = sprite;
    if (userData.shownAt !== undefined && material.opacity < 1) {
      material.opacity = Math.min(1, (now - userData.shownAt) / FADE_MS);
    }
  }
  renderer.render(scene, camera);
});

function sphereOf(ids) {
  const center = new THREE.Vector3();
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (!ids.has(assigned[i])) continue;
    center.x += positions[i * 3];
    center.y += positions[i * 3 + 1];
    center.z += positions[i * 3 + 2];
    count++;
  }
  if (!count) return whole;
  center.divideScalar(count);
  let radius = 0;
  const point = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    if (!ids.has(assigned[i])) continue;
    radius = Math.max(radius, point.fromArray(positions, i * 3).distanceTo(center));
  }
  return new THREE.Sphere(center, radius);
}

// what is shown
let fadeStart = -Infinity;

function recolour() {
  const ids = focusIds();
  const from = geometry.attributes.colorFrom.array;
  const to = geometry.attributes.colorTo.array;
  // start from where a fade still running has got to
  const k = pointsMaterial.uniforms.fade.value;
  for (let i = 0; i < n * 3; i++) from[i] = from[i] + (to[i] - from[i]) * k;
  for (let i = 0; i < n; i++) {
    const label = assigned[i];
    const colour =
      label < 0 ? NOISE : ids === null || ids.has(label) ? colours.get(label) : DIM;
    to.set(colour, i * 3);
  }
  geometry.attributes.colorFrom.needsUpdate = true;
  geometry.attributes.colorTo.needsUpdate = true;
  fadeStart = performance.now();
  pointsMaterial.uniforms.fade.value = 0;
}

function reindex() {
  const rows = [];
  for (let i = 0; i < n; i++) if (!hidden.has(assigned[i])) rows.push(i);
  geometry.setIndex(rows);
}

// one sprite per row, made once: the same rows (and URLs) come back every time
const loader = new THREE.TextureLoader();
const spriteCache = new Map();
const spriteSize = whole.radius * SPRITE_SIZE;
function spriteFor(row) {
  if (!spriteCache.has(row)) {
    const sprite = new THREE.Sprite();
    sprite.material = new THREE.SpriteMaterial({
      map: loader.load(`thumb/${row}?size=128`, (map) => {
        const { width, height } = map.image;
        sprite.scale.set((spriteSize * width) / height, spriteSize, 1);
        sprite.userData.shownAt = performance.now();
      }),
      opacity: 0,
    });
    sprite.scale.set(spriteSize, spriteSize, 1);
    sprite.position.fromArray(positions, row * 3);
    spriteCache.set(row, sprite);
  }
  return spriteCache.get(row);
}

// at least the most typical image of every cluster, then more while there is room
function spriteRows() {
  const ids = focusIds();
  const shown = info.clusters.filter(
    (c) => !hidden.has(c.id) && (ids === null || ids.has(c.id)),
  );
  const [depth, room] = ids === null ? [3, 300] : [12, 60];
  const rows = shown.filter((c) => c.samples.length).map((c) => c.samples[0]);
  for (let d = 1; d < depth; d++) {
    for (const cluster of shown) {
      if (rows.length >= room) return rows;
      if (d < cluster.samples.length) rows.push(cluster.samples[d]);
    }
  }
  return rows;
}

function placeSprites() {
  const before = new Set(sprites.children);
  sprites.clear();
  if (!showImages) return;
  const now = performance.now();
  for (const sprite of spriteRows().map(spriteFor)) {
    // a loaded image that comes into view fades in; one that stays, stays
    if (!before.has(sprite) && sprite.userData.shownAt !== undefined) {
      sprite.material.opacity = 0;
      sprite.userData.shownAt = now;
    }
    sprites.add(sprite);
  }
}

// names, shared by the list and the pane
const nameInputs = new Map(); // cluster id -> its input in the list

async function saveName(cluster, value, state) {
  state.textContent = "…";
  const response = await fetch(`clusters/${cluster.id}/name`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: value }),
  });
  if (!response.ok) return failed(state, await response.text());
  cluster.name = (await response.json()).name;
  nameInputs.get(cluster.id).value = cluster.name;
  if (focus?.cluster === cluster.id) paneName.value = cluster.name;
  renderGroups();
  saved(state);
}

function saved(state) {
  state.textContent = "✓";
  state.className = "state w-4 text-xs text-green-600";
  state.title = "";
  setTimeout(() => (state.textContent = ""), 1500);
}

function failed(state, why) {
  state.textContent = "!";
  state.className = "state w-4 text-xs text-red-600";
  state.title = why;
}

function nameInput(input, onSave) {
  input.addEventListener("click", (event) => event.stopPropagation());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") input.blur();
  });
  input.addEventListener("change", () => onSave(input.value));
}

// groups
async function putGroup(name, members, rename = name) {
  const response = await fetch(`groups/${encodeURIComponent(name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clusters: [...members], name: rename }),
  });
  if (!response.ok) {
    alert(await response.text());
    return false;
  }
  const group = await response.json();
  delete groups[name];
  groups[group.name] = group.clusters;
  if (focus?.group === name) focus = { group: group.name };
  refresh();
  return true;
}

async function deleteGroup(name) {
  const response = await fetch(`groups/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!response.ok) return alert(await response.text());
  delete groups[name];
  if (focus?.group === name) setFocus(null);
  else refresh();
}

const groupNames = () => Object.keys(groups).sort((a, b) => a.localeCompare(b));
const groupsOf = (id) => groupNames().filter((name) => groups[name].includes(id));
const imagesIn = (ids) => ids.reduce((sum, id) => sum + sizeOf(id), 0);
const label = (id) => {
  const name = clusters.get(id)?.name;
  return name ? `#${id} ${name}` : `#${id}`;
};

// the sidebar
const groupList = document.getElementById("groups");
function renderGroups() {
  groupList.replaceChildren(
    ...groupNames().map((name) => {
      const item = element(
        "li",
        "px-3 py-2 cursor-pointer flex items-baseline gap-2 hover:bg-neutral-50",
      );
      item.dataset.group = name;
      item.append(
        element("span", "flex-1 text-sm font-medium truncate", name),
        element(
          "span",
          "text-xs text-neutral-500 tabular-nums",
          `${groups[name].length} clusters · ${imagesIn(groups[name]).toLocaleString()}`,
        ),
      );
      item.addEventListener("click", () =>
        setFocus(focus?.group === name ? null : { group: name }),
      );
      return item;
    }),
  );
  document.getElementById("no-groups").classList.toggle("hidden", groupNames().length > 0);
  filter();
  highlight();
}

function thumbnailLink(row, className) {
  const link = element("a", "shrink-0");
  link.href = `thumb/${row}?size=512`;
  link.target = "_blank";
  const image = element("img", className);
  image.src = `thumb/${row}?size=128`;
  image.loading = "lazy";
  link.append(image);
  return link;
}

const ICON = 'class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
const EYE = `<svg ${ICON}><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>`;
const EYE_OFF = `<svg ${ICON}><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.4 10.4 0 0 1 12 5c7 0 10 7 10 7a13 13 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.5 13.5 0 0 0 2 12s3 7 10 7a9.7 9.7 0 0 0 5.39-1.61"/><line x1="2" x2="22" y1="2" y2="22"/></svg>`;

// the eye of a row: hides or shows that cluster (or the noise) in the view
function eye(button, id, item) {
  const draw = () => {
    button.innerHTML = hidden.has(id) ? EYE_OFF : EYE;
    item.classList.toggle("opacity-50", hidden.has(id));
  };
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    if (hidden.has(id)) hidden.delete(id);
    else hidden.add(id);
    draw();
    reindex();
    placeSprites();
  });
  draw();
}

const list = document.getElementById("clusters");
const template = document.getElementById("cluster-item");
const items = new Map();
const noiseCount = element("span", "text-xs tabular-nums");
if (info.noise) {
  const item = element("li", "px-3 py-2 flex items-center gap-2 text-sm text-neutral-500");
  const button = element("button", "eye shrink-0 text-neutral-500 hover:text-neutral-900");
  button.title = "show or hide in the view";
  const swatch = element("span", "w-3 h-3 rounded-full shrink-0");
  swatch.style.background = css(NOISE);
  // the same width as a checkbox, so the eyes line up
  item.append(
    element("span", "w-[13px] shrink-0"),
    button,
    swatch,
    element("span", "flex-1", "noise"),
    noiseCount,
  );
  eye(button, -1, item);
  list.append(item);
}
for (const cluster of [...info.clusters].sort((a, b) => a.id - b.id)) {
  const item = template.content.firstElementChild.cloneNode(true);
  eye(item.querySelector(".eye"), cluster.id, item);
  item.querySelector(".swatch").style.background = css(colours.get(cluster.id));
  item.querySelector(".id").textContent = `#${cluster.id}`;
  item.querySelector(".select").addEventListener("click", () =>
    setFocus(focus?.cluster === cluster.id ? null : { cluster: cluster.id }),
  );

  const check = item.querySelector(".check");
  check.addEventListener("click", (event) => event.stopPropagation());
  check.addEventListener("change", () => {
    if (check.checked) checked.add(cluster.id);
    else checked.delete(cluster.id);
    renderSelection();
  });

  const name = item.querySelector(".name");
  const state = item.querySelector(".state");
  name.value = cluster.name;
  nameInputs.set(cluster.id, name);
  nameInput(name, (value) => saveName(cluster, value, state));

  const details = item.querySelector("details");
  details.addEventListener("toggle", () => {
    const grid = item.querySelector(".samples");
    if (!details.open || grid.childElementCount) return;
    grid.append(
      ...cluster.samples
        .slice(0, LIST_SAMPLES)
        .map((row) => thumbnailLink(row, "w-full aspect-square object-cover rounded")),
    );
  });

  items.set(cluster.id, item);
  list.append(item);
}

const filterInput = document.getElementById("filter");
function filter() {
  const query = filterInput.value.trim().toLowerCase();
  const matches = (text) => !query || text.toLowerCase().includes(query);
  const viaGroup = new Set(
    groupNames().filter((name) => query && matches(name)).flatMap((name) => groups[name]),
  );
  items.forEach((item, id) => {
    const shown = matches(`#${id}`) || matches(clusters.get(id).name) || viaGroup.has(id);
    item.classList.toggle("hidden", !shown);
  });
  groupList.querySelectorAll("li").forEach((item) =>
    item.classList.toggle("hidden", !matches(item.dataset.group)),
  );
}
filterInput.addEventListener("input", filter);

function highlight() {
  const ids = focusIds();
  items.forEach((item, id) => item.classList.toggle("bg-amber-50", ids?.has(id) ?? false));
  groupList.querySelectorAll("li").forEach((item) =>
    item.classList.toggle("bg-amber-50", focus?.group === item.dataset.group),
  );
}

// multi-select and the group picker
const selectionBar = document.getElementById("selection-bar");
const picker = document.getElementById("group-picker");
const groupSearch = document.getElementById("group-search");
const groupOptions = document.getElementById("group-options");

function renderSelection() {
  selectionBar.classList.toggle("hidden", checked.size === 0);
  document.getElementById("selection-count").textContent =
    `${checked.size} cluster${checked.size === 1 ? "" : "s"} selected`;
  if (checked.size === 0) picker.classList.add("hidden");
}

function clearSelection() {
  checked.clear();
  items.forEach((item) => (item.querySelector(".check").checked = false));
  renderSelection();
}

async function addToGroup(name) {
  const members = new Set([...(groups[name] ?? []), ...checked]);
  if (await putGroup(name, members)) {
    clearSelection();
    setFocus({ group: name });
  }
}

function renderOptions() {
  const query = groupSearch.value.trim();
  const options = groupNames()
    .filter((name) => name.toLowerCase().includes(query.toLowerCase()))
    .map((name) => {
      const option = element(
        "li",
        "px-2 py-1 rounded cursor-pointer hover:bg-neutral-100 flex gap-2",
      );
      option.append(
        element("span", "flex-1 truncate", name),
        element("span", "text-xs text-neutral-500", `${groups[name].length} clusters`),
      );
      option.addEventListener("click", () => addToGroup(name));
      return option;
    });
  if (query && !(query in groups)) {
    const create = element(
      "li",
      "px-2 py-1 rounded cursor-pointer hover:bg-neutral-100 text-blue-700",
      `create "${query}"`,
    );
    create.addEventListener("click", () => addToGroup(query));
    options.push(create);
  }
  groupOptions.replaceChildren(...options);
}

document.getElementById("add-group").addEventListener("click", (event) => {
  event.stopPropagation();
  picker.classList.toggle("hidden");
  if (!picker.classList.contains("hidden")) {
    groupSearch.value = "";
    renderOptions();
    groupSearch.focus();
  }
});
groupSearch.addEventListener("input", renderOptions);
groupSearch.addEventListener("keydown", (event) => {
  const query = groupSearch.value.trim();
  if (event.key === "Enter" && query) addToGroup(query);
});
picker.addEventListener("click", (event) => event.stopPropagation());
document.addEventListener("click", () => picker.classList.add("hidden"));
document.getElementById("clear-selection").addEventListener("click", clearSelection);

// the bottom pane
const paneName = document.getElementById("pane-name");
const paneState = document.getElementById("pane-state");
const paneFacts = document.getElementById("pane-facts");
const paneSamples = document.getElementById("pane-samples");
const deleteButton = document.getElementById("delete-group");

nameInput(paneName, (value) => {
  if ("cluster" in focus) return saveName(clusters.get(focus.cluster), value, paneState);
  const name = value.trim();
  if (!name || name === focus.group) {
    paneName.value = focus.group;
    return;
  }
  putGroup(focus.group, groups[focus.group], name);
});
deleteButton.addEventListener("click", () => {
  if (confirm(`Delete the group "${focus.group}"? Its clusters stay.`)) {
    deleteGroup(focus.group);
  }
});

const fact = (name, value) => {
  const node = element("span", "", `${name} `);
  node.append(element("span", "font-medium text-neutral-900", value));
  return node;
};

function chip(text, onClick, onRemove = null) {
  const node = element(
    "span",
    "inline-flex items-center gap-1 rounded-full bg-neutral-100 border border-neutral-200 px-2 py-0.5 cursor-pointer hover:bg-neutral-200",
  );
  node.append(element("span", "", text));
  node.addEventListener("click", onClick);
  if (onRemove) {
    const remove = element("button", "text-neutral-500 hover:text-red-600", "×");
    remove.title = "remove from the group";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      onRemove();
    });
    node.append(remove);
  }
  return node;
}

const sampleClass = "h-full max-h-40 aspect-square object-cover rounded";
const hardIn = (ids) => ids.reduce((sum, id) => sum + (clusters.get(id)?.size ?? 0), 0);

// the soft threshold of the focused cluster: its own, or the global one
const paneSoft = document.getElementById("pane-soft");
const paneSlider = document.getElementById("pane-soft-slider");
const paneSoftValue = document.getElementById("pane-soft-value");
const paneSoftReset = document.getElementById("pane-soft-reset");

function showPaneSoft() {
  const id = focus?.cluster;
  if (id === undefined) return;
  const own = soft.clusters[id];
  paneSoftValue.textContent =
    own !== undefined
      ? number(own)
      : soft.threshold === null
        ? "global (off)"
        : `global (${number(soft.threshold)})`;
  // invisible, not gone: the row keeps its size
  paneSoftReset.classList.toggle("invisible", own === undefined);
}

paneSlider.addEventListener("input", () => {
  soft.clusters[focus.cluster] = Number(paneSlider.value);
  applySoft();
});
paneSlider.addEventListener("change", saveSoft);
paneSoftReset.addEventListener("click", () => {
  delete soft.clusters[focus.cluster];
  paneSlider.value = soft.threshold ?? 0.5;
  applySoft();
  saveSoft();
});

// the images soft thresholding added, the ones added last (lowest score) first
const paneJoined = document.getElementById("pane-joined");
const paneJoinedCaption = document.getElementById("pane-joined-caption");
const paneJoinedSamples = document.getElementById("pane-joined-samples");
let joinedFrame = null;

function renderJoined() {
  joinedFrame = null;
  const ids = focusIds();
  const rows = [];
  if (ids !== null && softLabels !== null) {
    for (let i = 0; i < n; i++) if (joined[i] && ids.has(assigned[i])) rows.push(i);
  }
  paneJoined.classList.toggle("hidden", rows.length === 0);
  if (!rows.length) return;
  rows.sort((a, b) => softScores[a] - softScores[b]);
  paneJoinedCaption.textContent =
    `+${rows.length.toLocaleString()} added from noise by soft thresholding, ` +
    "lowest score first";
  paneJoinedSamples.replaceChildren(
    ...rows.slice(0, PANE_SAMPLES).map((row) => {
      const link = thumbnailLink(row, sampleClass);
      link.title = `score ${number(softScores[row])}`;
      return link;
    }),
  );
}

// at most once a frame, while a slider moves
const scheduleJoined = () => {
  joinedFrame ??= requestAnimationFrame(renderJoined);
};

function renderPane() {
  const open = focus !== null;
  const wasOpen = !pane.classList.contains("hidden");
  pane.classList.toggle("hidden", !open);
  if (open !== wasOpen) resize();
  if (!open) return;

  paneState.textContent = "";
  if ("cluster" in focus) {
    const cluster = clusters.get(focus.cluster);
    const { stats } = cluster;
    document.getElementById("pane-swatch").style.background = css(colours.get(cluster.id));
    document.getElementById("pane-kind").textContent = `cluster #${cluster.id}`;
    paneName.value = cluster.name;
    paneName.placeholder = "unnamed";
    deleteButton.classList.add("hidden");
    paneSoft.classList.toggle("hidden", !info.has_soft);
    paneSlider.value = thresholdOf(cluster.id) ?? 0.5;
    showPaneSoft();
    paneFacts.replaceChildren(
      fact("images", cluster.size.toLocaleString()),
      fact("share", `${number(stats.share, 1)} %`),
      fact("persistence", number(stats.persistence)),
      fact("mean probability", number(stats.probability)),
      fact("median outlier score", number(stats.outlier)),
      fact("spread", number(stats.spread)),
      ...groupsOf(cluster.id).map((name) =>
        chip(`group: ${name}`, () => setFocus({ group: name })),
      ),
    );
    paneSamples.replaceChildren(
      ...cluster.samples.slice(0, PANE_SAMPLES).map((row) => thumbnailLink(row, sampleClass)),
    );
  } else {
    const name = focus.group;
    const members = groups[name];
    document.getElementById("pane-swatch").style.background = "transparent";
    document.getElementById("pane-kind").textContent = "group";
    paneName.value = name;
    paneName.placeholder = "group name";
    deleteButton.classList.remove("hidden");
    paneSoft.classList.add("hidden");
    paneFacts.replaceChildren(
      fact("clusters", members.length.toString()),
      fact("images", hardIn(members).toLocaleString()),
      ...members.map((id) =>
        chip(
          label(id),
          () => setFocus({ cluster: id }),
          () => putGroup(name, members.filter((m) => m !== id)),
        ),
      ),
    );
    const rows = members.flatMap((id) => clusters.get(id).samples.slice(0, 3));
    paneSamples.replaceChildren(
      ...rows.slice(0, PANE_SAMPLES).map((row) => thumbnailLink(row, sampleClass)),
    );
  }
  renderJoined();
}

// soft assignment: apply the thresholds, show the result, save on release
function counts() {
  items.forEach((item, id) => (item.querySelector(".size").textContent = countText(id)));
  noiseCount.textContent = noiseLeft().toLocaleString();
  showPaneSoft();
  if (focus) scheduleJoined();
  renderGroups();
}

function applySoft() {
  assign();
  const scale = geometry.attributes.scale.array;
  for (let i = 0; i < n; i++) scale[i] = joined[i] ? SOFT_SCALE : 1;
  geometry.attributes.scale.needsUpdate = true;
  recolour();
  reindex();
  counts();
}

async function saveSoft() {
  const response = await fetch("soft", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(soft),
  });
  if (!response.ok) alert(`Could not save the soft thresholds: ${await response.text()}`);
}

if (info.has_soft) {
  const on = document.getElementById("soft-on");
  const slider = document.getElementById("soft-threshold");
  const value = document.getElementById("soft-value");
  document.getElementById("soft-controls").classList.remove("hidden");
  if (soft.threshold !== null) slider.value = soft.threshold;
  const show = () => {
    on.checked = soft.threshold !== null;
    slider.disabled = !on.checked;
    value.textContent = on.checked ? number(soft.threshold) : "off";
  };
  on.addEventListener("change", () => {
    soft.threshold = on.checked ? Number(slider.value) : null;
    show();
    applySoft();
    saveSoft();
  });
  slider.addEventListener("input", () => {
    soft.threshold = Number(slider.value);
    show();
    applySoft();
  });
  slider.addEventListener("change", saveSoft);
  show();
}

// everything that depends on the focus or the groups
function refresh() {
  renderGroups();
  recolour();
  placeSprites();
  renderPane();
}

function setFocus(next) {
  focus = next;
  refresh();
  const ids = focusIds();
  if (ids === null) flyTo(whole, false);
  else flyTo(sphereOf(ids), true);
  if (focus && "cluster" in focus) {
    items.get(focus.cluster).scrollIntoView({ block: "nearest" });
  }
}

// controls
document.getElementById("show-images").addEventListener("change", (event) => {
  showImages = event.target.checked;
  placeSprites();
});
document.getElementById("deselect").addEventListener("click", () => setFocus(null));
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!picker.classList.contains("hidden")) picker.classList.add("hidden");
  else if (document.activeElement.tagName !== "INPUT") setFocus(null);
});

// hover and click on points
const tooltip = document.getElementById("tooltip");
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = pointSize;
const pointer = new THREE.Vector2();
let hovered = null;
let pending = false;
let pressed = null;

function rowAt(event) {
  const box = renderer.domElement.getBoundingClientRect();
  pointer.set(
    ((event.clientX - box.left) / box.width) * 2 - 1,
    -((event.clientY - box.top) / box.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(cloud)[0];
  return hit ? hit.index : null;
}

renderer.domElement.addEventListener("pointermove", (event) => {
  if (pending || event.buttons) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    const row = rowAt(event);
    if (row === null) {
      tooltip.classList.add("hidden");
      hovered = null;
      return;
    }
    if (row !== hovered) {
      hovered = row;
      const image = element("img");
      image.src = `thumb/${row}?size=256`;
      tooltip.replaceChildren(
        image,
        element(
          "div",
          "mt-1",
          assigned[row] < 0
            ? "noise"
            : `${label(assigned[row])}${joined[row] ? ` · soft (${number(softScores[row])})` : ""}`,
        ),
      );
    }
    const box = view.getBoundingClientRect();
    tooltip.style.left = `${event.clientX - box.left + 14}px`;
    tooltip.style.top = `${event.clientY - box.top + 14}px`;
    tooltip.classList.remove("hidden");
  });
});
renderer.domElement.addEventListener("pointerleave", () =>
  tooltip.classList.add("hidden"),
);
renderer.domElement.addEventListener("pointerdown", (event) => {
  pressed = [event.clientX, event.clientY];
});
renderer.domElement.addEventListener("pointerup", (event) => {
  if (!pressed) return;
  const moved = Math.hypot(event.clientX - pressed[0], event.clientY - pressed[1]);
  pressed = null;
  if (moved > 4) return;
  const row = rowAt(event);
  if (row === null || assigned[row] < 0) return;
  setFocus({ cluster: assigned[row] });
});

applySoft();
refresh();
