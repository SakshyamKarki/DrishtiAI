const InputField = ({
  label,
  type = "text",
  placeholder,
  registration,
  error,
}) => (
  <div className="mb-5">
    <label className="block text-xs font-medium text-slate-400 mb-2 tracking-wide">
      {label}
    </label>
    <input
      type={type}
      placeholder={placeholder}
      {...registration}
      className="w-full rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-all"
      style={{
        background: error ? "rgba(248,113,113,0.05)" : "rgba(255,255,255,0.05)",
        border: error
          ? "0.5px solid rgba(248,113,113,0.5)"
          : "0.5px solid rgba(255,255,255,0.1)",
      }}
    />
    {error && <p className="text-xs text-red-400 mt-1.5">{error.message}</p>}
  </div>
);

export default InputField;
