import EyeBackground from "../components/EyeBackground";

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-transparent">
      {/* ── Animated background ── */}
      {/* <EyeBackground /> */}

      {/* ── Your page content sits on top ── */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen text-white px-4">
        <h1 className="text-6xl font-bold tracking-tight mb-4 drop-shadow-lg">
          Dishti AI
        </h1>
        <p className="text-lg text-white/60 max-w-md text-center">
          Detect generated face from real.
        </p>
        <button className="mt-8 px-8 py-3 bg-blue-600 hover:bg-blue-500 rounded-full text-sm font-semibold tracking-wide transition-colors">
          DETECT
        </button>
      </div>
    </div>
  );
}
