var m = Object.defineProperty;
var v = (i, e, r) => e in i ? m(i, e, { enumerable: !0, configurable: !0, writable: !0, value: r }) : i[e] = r;
var o = (i, e, r) => v(i, typeof e != "symbol" ? e + "" : e, r);
function h(i, e, r) {
  const t = i.createShader(e);
  if (!t) throw new Error("createShader failed");
  if (i.shaderSource(t, r), i.compileShader(t), !i.getShaderParameter(t, i.COMPILE_STATUS)) {
    const s = i.getShaderInfoLog(t) ?? "";
    throw i.deleteShader(t), new Error(`Shader compile failed: ${s}`);
  }
  return t;
}
function g(i, e, r) {
  const t = i.createProgram();
  if (!t) throw new Error("createProgram failed");
  if (i.attachShader(t, e), i.attachShader(t, r), i.linkProgram(t), !i.getProgramParameter(t, i.LINK_STATUS)) {
    const s = i.getProgramInfoLog(t) ?? "";
    throw i.deleteProgram(t), new Error(`Program link failed: ${s}`);
  }
  return t;
}
function b(i) {
  const e = i.createBuffer();
  if (!e) throw new Error("createBuffer failed");
  return i.bindBuffer(i.ARRAY_BUFFER, e), i.bufferData(
    i.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    i.STATIC_DRAW
  ), e;
}
function f(i) {
  const e = document.createElement("div");
  e.style.color = i, e.style.display = "none", document.body.appendChild(e);
  const r = getComputedStyle(e).color;
  document.body.removeChild(e);
  const t = r.match(/rgba?\(([^)]+)\)/);
  if (!t || !t[1]) return [0, 0, 0];
  const s = t[1].split(",").map((a) => parseFloat(a.trim()));
  return [(s[0] ?? 0) / 255, (s[1] ?? 0) / 255, (s[2] ?? 0) / 255];
}
const R = `
attribute vec2 aPos;
void main() {
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`, S = `
precision highp float;

uniform vec2  uResolution;
uniform float uTime;
uniform float uCellSize;
uniform float uDotRadius;
uniform vec3  uBgColor;
uniform vec3  uDotColor;
uniform float uNoiseScale;
uniform float uNoiseSpeed;
uniform float uNoiseContrast;
uniform float uNoiseBias;
uniform float uNoiseWarp;
uniform float uNoiseAnisotropy;
uniform float uNoiseAngle;
uniform float uSeed;
uniform vec2  uPointer;
uniform float uHoverRadius;
uniform float uHoverStrength;
uniform float uHoverFalloff;

// Bayer 8x8 thresholds (0..63) / 64
float bayer8(int x, int y) {
  int i = y * 8 + x;
  // unrolled lookup — WebGL1 safe
  if (i ==  0) return  0.0; if (i ==  1) return 32.0; if (i ==  2) return  8.0; if (i ==  3) return 40.0;
  if (i ==  4) return  2.0; if (i ==  5) return 34.0; if (i ==  6) return 10.0; if (i ==  7) return 42.0;
  if (i ==  8) return 48.0; if (i ==  9) return 16.0; if (i == 10) return 56.0; if (i == 11) return 24.0;
  if (i == 12) return 50.0; if (i == 13) return 18.0; if (i == 14) return 58.0; if (i == 15) return 26.0;
  if (i == 16) return 12.0; if (i == 17) return 44.0; if (i == 18) return  4.0; if (i == 19) return 36.0;
  if (i == 20) return 14.0; if (i == 21) return 46.0; if (i == 22) return  6.0; if (i == 23) return 38.0;
  if (i == 24) return 60.0; if (i == 25) return 28.0; if (i == 26) return 52.0; if (i == 27) return 20.0;
  if (i == 28) return 62.0; if (i == 29) return 30.0; if (i == 30) return 54.0; if (i == 31) return 22.0;
  if (i == 32) return  3.0; if (i == 33) return 35.0; if (i == 34) return 11.0; if (i == 35) return 43.0;
  if (i == 36) return  1.0; if (i == 37) return 33.0; if (i == 38) return  9.0; if (i == 39) return 41.0;
  if (i == 40) return 51.0; if (i == 41) return 19.0; if (i == 42) return 59.0; if (i == 43) return 27.0;
  if (i == 44) return 49.0; if (i == 45) return 17.0; if (i == 46) return 57.0; if (i == 47) return 25.0;
  if (i == 48) return 15.0; if (i == 49) return 47.0; if (i == 50) return  7.0; if (i == 51) return 39.0;
  if (i == 52) return 13.0; if (i == 53) return 45.0; if (i == 54) return  5.0; if (i == 55) return 37.0;
  if (i == 56) return 63.0; if (i == 57) return 31.0; if (i == 58) return 55.0; if (i == 59) return 23.0;
  if (i == 60) return 61.0; if (i == 61) return 29.0; if (i == 62) return 53.0; return 21.0;
}

float hash(vec2 p) {
  p = fract(p * vec2(127.1, 311.7) + uSeed);
  return fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453);
}

float valueNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  for (int k = 0; k < 4; k++) {
    sum += amp * valueNoise(p);
    p *= 2.03;
    amp *= 0.5;
  }
  return sum;
}

int imod(int a, int b) {
  return a - (a / b) * b;
}

void main() {
  vec2 frag = gl_FragCoord.xy;
  vec2 cellCoord = floor(frag / uCellSize);
  vec2 cellCenter = (cellCoord + 0.5) * uCellSize;

  float ca = cos(uNoiseAngle);
  float sa = sin(uNoiseAngle);
  mat2 R = mat2(ca, -sa, sa, ca);
  vec2 q = R * (cellCenter * uNoiseScale);
  q.x /= (1.0 + uNoiseAnisotropy);
  q += vec2(uTime * uNoiseSpeed, -uTime * uNoiseSpeed * 0.6);
  vec2 w = vec2(fbm(q + vec2(13.7, 5.2)), fbm(q + vec2(91.3, 47.1)));
  float n = fbm(q + uNoiseWarp * w);
  n = clamp((n - 0.5) * uNoiseContrast + 0.5 + uNoiseBias, 0.0, 1.0);

  float d = distance(cellCenter, uPointer);
  float g = exp(-pow(max(d / max(uHoverRadius, 1.0), 0.0), uHoverFalloff));
  float mask = clamp(n + uHoverStrength * g, 0.0, 1.0);

  int bx = imod(int(cellCoord.x), 8);
  int by = imod(int(cellCoord.y), 8);
  float t = (bayer8(bx, by) + 0.5) / 64.0;
  float on = step(t, mask);

  vec2 local = (frag - cellCenter) / (uCellSize * 0.5);
  float aa = 1.5 / uCellSize;
  float disk = 1.0 - smoothstep(uDotRadius - aa, uDotRadius + aa, length(local));

  float alpha = on * disk;
  vec3 col = mix(uBgColor, uDotColor, alpha);
  gl_FragColor = vec4(col, 1.0);
}
`, x = {
  cellSize: 6,
  dotRadius: 0.35,
  colors: { background: "#0b0b0d", dot: "#a8a8b0" },
  noise: {
    scale: 4e-3,
    speed: 0.03,
    contrast: 2.2,
    bias: -0.2,
    warp: 2.5,
    anisotropy: 1.8,
    angle: -Math.PI / 4,
    seed: 0
  },
  hover: { radius: 120, strength: -1, falloff: 2 },
  pointerLerp: 0.18,
  hoverLerp: 0.12,
  dpr: "auto",
  paused: !1
};
function l(i, e) {
  return e ? {
    ...i,
    ...e,
    colors: { ...i.colors, ...e.colors ?? {} },
    noise: { ...i.noise, ...e.noise ?? {} },
    hover: { ...i.hover, ...e.hover ?? {} }
  } : i;
}
class A {
  constructor(e, r) {
    o(this, "host");
    o(this, "canvas");
    o(this, "gl");
    o(this, "program");
    o(this, "quad");
    o(this, "aPos");
    o(this, "uniforms", {});
    o(this, "opts");
    o(this, "bgRgb");
    o(this, "dotRgb");
    o(this, "cssW", 0);
    o(this, "cssH", 0);
    o(this, "pxW", 0);
    o(this, "pxH", 0);
    o(this, "pointer", { x: -9999, y: -9999 });
    o(this, "pointerTarget", { x: -9999, y: -9999 });
    o(this, "hoverActive", 0);
    o(this, "hoverTarget", 0);
    o(this, "startTime", performance.now());
    o(this, "rafId", 0);
    o(this, "running", !1);
    o(this, "ro", null);
    o(this, "onMove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointerTarget.x = e.clientX - r.left, this.pointerTarget.y = e.clientY - r.top, this.hoverTarget = 1;
    });
    o(this, "onLeave", () => {
      this.hoverTarget = 0;
    });
    o(this, "onEnter", () => {
      this.hoverTarget = 1;
    });
    this.host = e, this.opts = l(x, r), this.bgRgb = f(this.opts.colors.background), this.dotRgb = f(this.opts.colors.dot), getComputedStyle(e).position === "static" && (e.style.position = "relative"), this.canvas = document.createElement("canvas"), Object.assign(this.canvas.style, {
      position: "absolute",
      inset: "0",
      width: "100%",
      height: "100%",
      display: "block",
      pointerEvents: "auto",
      touchAction: "none"
    }), e.appendChild(this.canvas);
    const t = this.canvas.getContext("webgl", { antialias: !1, premultipliedAlpha: !1, alpha: !1 });
    if (!t) throw new Error("Dithervas: WebGL not supported");
    this.gl = t;
    const s = h(t, t.VERTEX_SHADER, R), a = h(t, t.FRAGMENT_SHADER, S);
    this.program = g(t, s, a), t.deleteShader(s), t.deleteShader(a), this.quad = b(t), this.aPos = t.getAttribLocation(this.program, "aPos");
    for (const u of [
      "uResolution",
      "uTime",
      "uCellSize",
      "uDotRadius",
      "uBgColor",
      "uDotColor",
      "uNoiseScale",
      "uNoiseSpeed",
      "uNoiseContrast",
      "uNoiseBias",
      "uNoiseWarp",
      "uNoiseAnisotropy",
      "uNoiseAngle",
      "uSeed",
      "uPointer",
      "uHoverRadius",
      "uHoverStrength",
      "uHoverFalloff"
    ])
      this.uniforms[u] = t.getUniformLocation(this.program, u);
    this.ro = new ResizeObserver(() => this.resize()), this.ro.observe(e), this.resize(), this.canvas.addEventListener("pointermove", this.onMove), this.canvas.addEventListener("pointerenter", this.onEnter), this.canvas.addEventListener("pointerleave", this.onLeave), this.opts.paused || this.resume();
  }
  update(e) {
    this.opts = l(this.opts, e), e.colors?.background !== void 0 && (this.bgRgb = f(this.opts.colors.background)), e.colors?.dot !== void 0 && (this.dotRgb = f(this.opts.colors.dot)), e.dpr !== void 0 && this.resize();
  }
  pause() {
    this.running = !1, this.rafId && cancelAnimationFrame(this.rafId), this.rafId = 0;
  }
  resume() {
    if (this.running) return;
    this.running = !0;
    const e = () => {
      this.running && (this.frame(), this.rafId = requestAnimationFrame(e));
    };
    this.rafId = requestAnimationFrame(e);
  }
  destroy() {
    this.pause(), this.ro?.disconnect(), this.ro = null, this.canvas.removeEventListener("pointermove", this.onMove), this.canvas.removeEventListener("pointerenter", this.onEnter), this.canvas.removeEventListener("pointerleave", this.onLeave);
    const e = this.gl;
    e.deleteBuffer(this.quad), e.deleteProgram(this.program), this.canvas.remove();
  }
  resize() {
    const e = this.host.getBoundingClientRect(), r = this.opts.dpr === "auto" ? window.devicePixelRatio || 1 : this.opts.dpr, t = Math.min(Math.max(r, 1), 2);
    this.cssW = Math.max(1, e.width), this.cssH = Math.max(1, e.height), this.pxW = Math.round(this.cssW * t), this.pxH = Math.round(this.cssH * t), this.canvas.width = this.pxW, this.canvas.height = this.pxH;
  }
  frame() {
    const e = this.gl, r = this.opts, t = r.pointerLerp;
    this.pointer.x += (this.pointerTarget.x - this.pointer.x) * t, this.pointer.y += (this.pointerTarget.y - this.pointer.y) * t, this.hoverActive += (this.hoverTarget - this.hoverActive) * r.hoverLerp;
    const s = this.pxW / this.cssW, a = this.pxH / this.cssH, u = this.pointer.x * s, c = this.pxH - this.pointer.y * a, d = r.cellSize * Math.max(s, a), p = r.hover.radius * Math.max(s, a);
    e.viewport(0, 0, this.pxW, this.pxH), e.disable(e.DEPTH_TEST), e.disable(e.BLEND), e.useProgram(this.program), e.bindBuffer(e.ARRAY_BUFFER, this.quad), e.enableVertexAttribArray(this.aPos), e.vertexAttribPointer(this.aPos, 2, e.FLOAT, !1, 0, 0);
    const n = this.uniforms;
    e.uniform2f(n.uResolution, this.pxW, this.pxH), e.uniform1f(n.uTime, (performance.now() - this.startTime) / 1e3), e.uniform1f(n.uCellSize, d), e.uniform1f(n.uDotRadius, r.dotRadius), e.uniform3f(n.uBgColor, this.bgRgb[0], this.bgRgb[1], this.bgRgb[2]), e.uniform3f(n.uDotColor, this.dotRgb[0], this.dotRgb[1], this.dotRgb[2]), e.uniform1f(n.uNoiseScale, r.noise.scale), e.uniform1f(n.uNoiseSpeed, r.noise.speed), e.uniform1f(n.uNoiseContrast, r.noise.contrast), e.uniform1f(n.uNoiseBias, r.noise.bias), e.uniform1f(n.uNoiseWarp, r.noise.warp), e.uniform1f(n.uNoiseAnisotropy, r.noise.anisotropy), e.uniform1f(n.uNoiseAngle, r.noise.angle), e.uniform1f(n.uSeed, r.noise.seed), e.uniform2f(n.uPointer, u, c), e.uniform1f(n.uHoverRadius, p), e.uniform1f(n.uHoverStrength, r.hover.strength * this.hoverActive), e.uniform1f(n.uHoverFalloff, r.hover.falloff), e.drawArrays(e.TRIANGLE_STRIP, 0, 4);
  }
}
export {
  A as Dithervas
};
//# sourceMappingURL=dithervas.js.map
