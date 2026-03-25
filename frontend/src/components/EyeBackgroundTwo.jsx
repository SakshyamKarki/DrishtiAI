import { useEffect, useRef } from "react";

// ─── Tuning knobs ──────────────────────────────────────────────
const BG_COLOR       = "#07070d";
const EYE_RADIUS     = 130;       // sclera radius in px (responsive on small screens)
const PUPIL_TRAVEL   = 0.38;      // how far pupil can move (fraction of iris_r)
const BLINK_SPEED    = 0.018;     // lower = slower blink
const BLINK_HOLD     = 60;        // frames to hold closed (~1s at 60fps)
const BLINK_INTERVAL = 220;       // avg frames between blinks (~3.6s at 60fps)
// ───────────────────────────────────────────────────────────────

/** Draw one large eye centered at (cx, cy) with radius R. blinkT: 0=open, 1=closed */
function drawEye(ctx, cx, cy, R, mouse, blinkT, irisColor1, irisColor2) {
  const iris_r  = R * 0.56;
  const pupil_r = iris_r * 0.42;

  // pupil follows mouse
  const dx    = mouse.x - cx;
  const dy    = mouse.y - cy;
  const angle = Math.atan2(dy, dx);
  const dist  = Math.min(1, Math.hypot(dx, dy) / 600);
  const px = cx + Math.cos(angle) * iris_r * PUPIL_TRAVEL * dist;
  const py = cy + Math.sin(angle) * iris_r * PUPIL_TRAVEL * dist;

  // atmospheric glow
  const gGlow = ctx.createRadialGradient(cx, cy, R * 0.5, cx, cy, R * 2.6);
  gGlow.addColorStop(0, hexAlpha(irisColor1, 0.16));
  gGlow.addColorStop(0.5, hexAlpha(irisColor2, 0.08));
  gGlow.addColorStop(1, "rgba(0,0,0,0)");
  ctx.beginPath();
  ctx.arc(cx, cy, R * 2.6, 0, Math.PI * 2);
  ctx.fillStyle = gGlow;
  ctx.fill();

  // sclera
  const gSclera = ctx.createRadialGradient(cx - R * 0.15, cy - R * 0.15, 0, cx, cy, R);
  gSclera.addColorStop(0, "#f4f2ec");
  gSclera.addColorStop(0.78, "#dddbd3");
  gSclera.addColorStop(1, "#9a9890");
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.fillStyle = gSclera;
  ctx.fill();

  // iris
  const gIris = ctx.createRadialGradient(px - iris_r * 0.2, py - iris_r * 0.2, 0, px, py, iris_r);
  gIris.addColorStop(0,   lightenHex(irisColor1, 55));
  gIris.addColorStop(0.4, irisColor1);
  gIris.addColorStop(1,   irisColor2);
  ctx.beginPath();
  ctx.arc(px, py, iris_r, 0, Math.PI * 2);
  ctx.fillStyle = gIris;
  ctx.fill();

  // iris fibre lines
  ctx.save();
  ctx.beginPath();
  ctx.arc(px, py, iris_r, 0, Math.PI * 2);
  ctx.clip();
  for (let i = 0; i < 32; i++) {
    const a = (i / 32) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(px + Math.cos(a) * iris_r * 0.35, py + Math.sin(a) * iris_r * 0.35);
    ctx.lineTo(px + Math.cos(a) * iris_r * 0.97, py + Math.sin(a) * iris_r * 0.97);
    ctx.strokeStyle = "rgba(255,255,255,0.11)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.restore();

  // pupil
  const gPupil = ctx.createRadialGradient(px, py, 0, px, py, pupil_r);
  gPupil.addColorStop(0, "#12121a");
  gPupil.addColorStop(1, "#030308");
  ctx.beginPath();
  ctx.arc(px, py, pupil_r, 0, Math.PI * 2);
  ctx.fillStyle = gPupil;
  ctx.fill();

  // specular highlights
  ctx.beginPath();
  ctx.arc(px - iris_r * 0.16, py - iris_r * 0.20, iris_r * 0.11, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(px + iris_r * 0.14, py + iris_r * 0.13, iris_r * 0.055, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.38)";
  ctx.fill();

  // sclera ring
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 2;
  ctx.stroke();

  // ── EYELIDS ──
  // Upper lid sweeps down, lower lid sweeps up. They meet at cy when blinkT=1.
  if (blinkT > 0) {
    const lidTravel = R * blinkT; // pixels each lid has moved toward center

    ctx.save();
    // clip drawing to sclera circle
    ctx.beginPath();
    ctx.arc(cx, cy, R + 1, 0, Math.PI * 2);
    ctx.clip();

    // upper eyelid (moves down from top)
    ctx.beginPath();
    ctx.moveTo(cx - R - 2, cy - R - 2);
    ctx.lineTo(cx + R + 2, cy - R - 2);
    ctx.lineTo(cx + R + 2, cy - R + lidTravel);
    // gentle curve at lid edge
    ctx.quadraticCurveTo(cx, cy - R + lidTravel + R * 0.08, cx - R - 2, cy - R + lidTravel);
    ctx.closePath();
    ctx.fillStyle = BG_COLOR;
    ctx.fill();

    // lower eyelid (moves up from bottom)
    ctx.beginPath();
    ctx.moveTo(cx - R - 2, cy + R + 2);
    ctx.lineTo(cx + R + 2, cy + R + 2);
    ctx.lineTo(cx + R + 2, cy + R - lidTravel);
    ctx.quadraticCurveTo(cx, cy + R - lidTravel - R * 0.06, cx - R - 2, cy + R - lidTravel);
    ctx.closePath();
    ctx.fillStyle = BG_COLOR;
    ctx.fill();

    // subtle shadow crease on upper lid edge
    ctx.beginPath();
    ctx.moveTo(cx - R, cy - R + lidTravel);
    ctx.quadraticCurveTo(cx, cy - R + lidTravel + R * 0.08, cx + R, cy - R + lidTravel);
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.lineWidth = 3;
    ctx.stroke();

    ctx.restore();
  }
}

// ── color helpers ──────────────────────────────────────────────
function lightenHex(hex, amt) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgb(${Math.min(255,r+amt)},${Math.min(255,g+amt)},${Math.min(255,b+amt)})`;
}
function hexAlpha(hex, alpha) {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Blink state ────────────────────────────────────────────────
function createBlinkState(frameOffset = 0) {
  return {
    blinkT:     0,
    phase:      "waiting",   // "waiting" | "closing" | "holding" | "opening"
    holdCount:  0,
    frameCount: 0,
    nextBlink:  BLINK_INTERVAL + frameOffset,
  };
}

function tickBlink(b) {
  b.frameCount++;
  if (b.phase === "waiting") {
    if (b.frameCount >= b.nextBlink) b.phase = "closing";
  } else if (b.phase === "closing") {
    b.blinkT += BLINK_SPEED;
    if (b.blinkT >= 1) { b.blinkT = 1; b.phase = "holding"; b.holdCount = 0; }
  } else if (b.phase === "holding") {
    if (++b.holdCount >= BLINK_HOLD) b.phase = "opening";
  } else if (b.phase === "opening") {
    b.blinkT -= BLINK_SPEED;
    if (b.blinkT <= 0) {
      b.blinkT = 0;
      b.phase  = "waiting";
      b.frameCount = 0;
      b.nextBlink  = BLINK_INTERVAL + Math.round(Math.random() * 120 - 60);
    }
  }
}

// ── Component ─────────────────────────────────────────────────
export default function EyeBackgroundTwo() {
  const canvasRef = useRef(null);
  const mouseRef  = useRef({ x: -9999, y: -9999 });
  const blinkL    = useRef(createBlinkState(0));
  const blinkR    = useRef(createBlinkState(0)); // staggered start
  const rafRef    = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
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

      // responsive radius — shrinks on narrow screens
      const R   = Math.min(EYE_RADIUS, W * 0.13);
      const gap = R * 2.5;
      const lx  = W / 2 - gap;
      const rx  = W / 2 + gap;
      const ey  = H / 2;

      // background
      ctx.fillStyle = BG_COLOR;
      ctx.fillRect(0, 0, W, H);

      // ambient center glow
      const amb = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, W * 0.65);
      amb.addColorStop(0, "rgba(10,18,50,0.9)");
      amb.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = amb;
      ctx.fillRect(0, 0, W, H);

      // tick blink states
      tickBlink(blinkL.current);
      tickBlink(blinkR.current);

      // draw eyes
      drawEye(ctx, lx, ey, R, mouse, blinkL.current.blinkT, "#1a6bff", "#0a1060");
      drawEye(ctx, rx, ey, R, mouse, blinkR.current.blinkT, "#3b1aff", "#150a60");

      // vignette overlay
      const vig = ctx.createRadialGradient(W/2, H/2, H*0.22, W/2, H/2, H*0.88);
      vig.addColorStop(0, "rgba(0,0,0,0)");
      vig.addColorStop(1, "rgba(0,0,0,0.75)");
      ctx.fillStyle = vig;
      ctx.fillRect(0, 0, W, H);

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
