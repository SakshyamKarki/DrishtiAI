import { useEffect, useRef } from "react";

// ─── Tuning knobs ──────────────────────────────────────────────
const SMALL_EYE_COUNT   = 80;   // number of floating small eyes
const SMALL_RADIUS      = 18;   // half-size of small eyes
const PUPIL_TRAVEL      = 0.38; // how far pupil travels (0–1 of iris radius)
const BLINK_CHANCE      = 0.003;// per-frame probability any eye blinks
const BIG_EYE_RADIUS    = 110;  // giant center eye iris radius
const BIG_PUPIL_TRAVEL  = 0.40;
const BG_COLOR          = "#0a0a0f";
const IRIS_COLORS       = ["#1a6bff","#8b2be2","#ff3c5f","#00c9a7","#ffaa00"];
// ───────────────────────────────────────────────────────────────

function randomIrisColor() {
  return IRIS_COLORS[Math.floor(Math.random() * IRIS_COLORS.length)];
}

function createSmallEye(W, H) {
  return {
    x: Math.random() * W,
    y: Math.random() * H,
    r: SMALL_RADIUS * (0.7 + Math.random() * 0.8),
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    irisColor: randomIrisColor(),
    blinkT: 0,       // 0 = open, 1 = fully closed
    blinking: false,
    blinkDir: 1,
    rotation: Math.random() * Math.PI * 2,
  };
}

function drawSmallEye(ctx, eye, mouse) {
  const { x, y, r, irisColor, blinkT, rotation } = eye;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  // ── Sclera (white) ──
  ctx.beginPath();
  ctx.ellipse(0, 0, r, r * 0.62, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#e8e8e0";
  ctx.fill();

  // ── Blink clip ──
  // We draw a fill over the top/bottom proportional to blinkT
  const blinkH = r * 0.62 * blinkT;

  // ── Iris ──
  const iris_r = r * 0.44;
  const dx = mouse.x - x, dy = mouse.y - y;
  const angle = Math.atan2(dy, dx);
  const dist = Math.min(1, Math.hypot(dx, dy) / 300);
  const px = Math.cos(angle) * iris_r * PUPIL_TRAVEL * dist;
  const py = Math.sin(angle) * iris_r * PUPIL_TRAVEL * dist;

  // iris gradient
  const gIris = ctx.createRadialGradient(px, py, 0, px, py, iris_r);
  gIris.addColorStop(0, lighten(irisColor, 40));
  gIris.addColorStop(1, irisColor);

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(0, 0, r, r * 0.62, 0, 0, Math.PI * 2);
  ctx.clip();

  ctx.beginPath();
  ctx.arc(px, py, iris_r, 0, Math.PI * 2);
  ctx.fillStyle = gIris;
  ctx.fill();

  // pupil
  ctx.beginPath();
  ctx.arc(px, py, iris_r * 0.45, 0, Math.PI * 2);
  ctx.fillStyle = "#050508";
  ctx.fill();

  // highlight
  ctx.beginPath();
  ctx.arc(px - iris_r * 0.18, py - iris_r * 0.22, iris_r * 0.14, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.fill();

  ctx.restore();

  // ── Eyelid blink overlay ──
  if (blinkT > 0) {
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(0, 0, r, r * 0.62, 0, 0, Math.PI * 2);
    ctx.clip();
    // top lid
    ctx.fillStyle = darken(irisColor, 50);
    ctx.fillRect(-r, -r * 0.62, r * 2, blinkH + 2);
    // bottom lid
    ctx.fillRect(-r, r * 0.62 - blinkH - 2, r * 2, blinkH + 2);
    ctx.restore();
  }

  // ── Outline ──
  ctx.beginPath();
  ctx.ellipse(0, 0, r, r * 0.62, 0, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.restore();
}

function drawBigEye(ctx, cx, cy, mouse) {
  const R = BIG_EYE_RADIUS;
  const dx = mouse.x - cx, dy = mouse.y - cy;
  const angle = Math.atan2(dy, dx);
  const dist = Math.min(1, Math.hypot(dx, dy) / 500);
  const px = cx + Math.cos(angle) * R * BIG_PUPIL_TRAVEL * dist;
  const py = cy + Math.sin(angle) * R * BIG_PUPIL_TRAVEL * dist;

  // outer glow
  const gGlow = ctx.createRadialGradient(cx, cy, R * 0.8, cx, cy, R * 2.2);
  gGlow.addColorStop(0, "rgba(30,100,255,0.18)");
  gGlow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.beginPath();
  ctx.arc(cx, cy, R * 2.2, 0, Math.PI * 2);
  ctx.fillStyle = gGlow;
  ctx.fill();

  // sclera
  const gSclera = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  gSclera.addColorStop(0, "#f0efe8");
  gSclera.addColorStop(0.85, "#d8d8d0");
  gSclera.addColorStop(1, "#b0b0a8");
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fillStyle = gSclera;
  ctx.fill();

  // iris
  const iris_r = R * 0.56;
  const gIris = ctx.createRadialGradient(px, py, 0, px, py, iris_r);
  gIris.addColorStop(0, "#5599ff");
  gIris.addColorStop(0.5, "#1a50d0");
  gIris.addColorStop(1, "#0a1a6a");
  ctx.beginPath();
  ctx.arc(px, py, iris_r, 0, Math.PI * 2);
  ctx.fillStyle = gIris;
  ctx.fill();

  // iris texture lines
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(px + Math.cos(a) * iris_r * 0.4, py + Math.sin(a) * iris_r * 0.4);
    ctx.lineTo(px + Math.cos(a) * iris_r * 0.95, py + Math.sin(a) * iris_r * 0.95);
    ctx.strokeStyle = "rgba(100,170,255,0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // pupil
  const gPupil = ctx.createRadialGradient(px, py, 0, px, py, iris_r * 0.4);
  gPupil.addColorStop(0, "#0a0a12");
  gPupil.addColorStop(1, "#020205");
  ctx.beginPath();
  ctx.arc(px, py, iris_r * 0.4, 0, Math.PI * 2);
  ctx.fillStyle = gPupil;
  ctx.fill();

  // specular highlights
  ctx.beginPath();
  ctx.arc(px - iris_r * 0.15, py - iris_r * 0.18, iris_r * 0.1, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(px + iris_r * 0.12, py + iris_r * 0.12, iris_r * 0.05, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.fill();

  // ring
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(80,140,255,0.3)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

// ── color helpers ──────────────────────────────────────────────
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return [r,g,b];
}
function lighten(hex, amt) {
  const [r,g,b] = hexToRgb(hex);
  return `rgb(${Math.min(255,r+amt)},${Math.min(255,g+amt)},${Math.min(255,b+amt)})`;
}
function darken(hex, amt) {
  const [r,g,b] = hexToRgb(hex);
  return `rgb(${Math.max(0,r-amt)},${Math.max(0,g-amt)},${Math.max(0,b-amt)})`;
}

// ── Main component ─────────────────────────────────────────────
export default function EyeBackground() {
  const canvasRef = useRef(null);
  const mouseRef  = useRef({ x: -9999, y: -9999 });
  const eyesRef   = useRef([]);
  const rafRef    = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      // re-scatter eyes on resize
      eyesRef.current = Array.from({ length: SMALL_EYE_COUNT }, () =>
        createSmallEye(canvas.width, canvas.height)
      );
    };

    const onMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMouseMove);
    resize();

    const animate = () => {
      const W = canvas.width, H = canvas.height;
      const mouse = mouseRef.current;

      // background
      ctx.fillStyle = BG_COLOR;
      ctx.fillRect(0, 0, W, H);

      // subtle vignette
      const vig = ctx.createRadialGradient(W/2, H/2, H*0.2, W/2, H/2, H*0.9);
      vig.addColorStop(0, "rgba(0,0,0,0)");
      vig.addColorStop(1, "rgba(0,0,0,0.65)");
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, W, H);

      // ── update + draw small eyes ──
      eyesRef.current.forEach((eye) => {
        // drift
        eye.x += eye.vx;
        eye.y += eye.vy;
        // wrap
        if (eye.x < -eye.r*2) eye.x = W + eye.r;
        if (eye.x > W + eye.r*2) eye.x = -eye.r;
        if (eye.y < -eye.r*2) eye.y = H + eye.r;
        if (eye.y > H + eye.r*2) eye.y = -eye.r;

        // blink logic
        if (!eye.blinking && Math.random() < BLINK_CHANCE) {
          eye.blinking = true;
          eye.blinkDir = 1;
        }
        if (eye.blinking) {
          eye.blinkT += eye.blinkDir * 0.12;
          if (eye.blinkT >= 1) { eye.blinkDir = -1; }
          if (eye.blinkT <= 0) { eye.blinkT = 0; eye.blinking = false; }
        }

        // eyes near cursor react faster (jiggle)
        const dist = Math.hypot(mouse.x - eye.x, mouse.y - eye.y);
        if (dist < 120) {
          eye.rotation += 0.015;
        }

        drawSmallEye(ctx, eye, mouse);
      });

      // ── draw big center eye ──
      drawBigEye(ctx, W / 2, H / 2, mouse);

      rafRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 w-full h-full"
      style={{ zIndex: 0 }}
    />
  );
}
