import { useDispatch, useSelector } from "react-redux";
import { Link, useNavigate } from "react-router-dom";
import { registerThunk, selectAuthLoading } from "../features/auth/authSlice";
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import registerSchema from "../validations/registerSchema";
import "../styles/auth.css";
import { useEffect } from "react";
import InputField from "../components/ui/InputField";

function RegisterPage() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const isLoading = useSelector(selectAuthLoading);
  const isAuthenticated = useSelector(selectAuthLoading);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(registerSchema),
  });

  useEffect(() => {
    if (isAuthenticated) navigate("/");
  }, [isAuthenticated]);

  const onSubmit = async (data) => {
    const { confirmPassword, ...registerData } = data;
    const result = await dispatch(registerThunk(registerData));
    if (registerThunk.fulfilled.match(result)) {
      navigate("/");
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-orb-1" />
      <div className="auth-orb-2" />

      <div className="auth-grid" />

      <div className="auth-card">
        <div className="auth-logo-row">
          <div className="auth-logo-ring">
            <div className="auth-logo-dot" />
          </div>
          <span className="auth-logo-name">
            Drishti<span className="text-indigo-400">AI</span>
          </span>
        </div>
        <div className="auth-badge">
          <span className="auth-badge-dot" />
          Create your account
        </div>
        <h1
          className="text-2xl font-semibold text-slate-100 mb-1"
          style={{ fontFamily: "Syne, sans-serif" }}
        >
          Get started
        </h1>
        <p className="text-sm text-slate-500 mb-8">
          Create your account to start detecting fake images
        </p>

        <form onSubmit={handleSubmit(onSubmit)}>
          <InputField
            label="username"
            registration={register("username")}
            error={errors.username}
          />

          <InputField
            label="Email address"
            type="email"
            placeholder="you@example.com"
            registration={register("email")}
            error={errors.email}
          />

          <InputField
            label="Password"
            type="password"
            placeholder="••••••••"
            registration={register("password")}
            error={errors.password}
          />

          <InputField
            label="Confirm password"
            type="password"
            placeholder="••••••••"
            registration={register("confirmPassword")}
            error={errors.confirmPassword}
          />

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-60 mt-2"
            style={{
              background: "linear-gradient(135deg, #6366f1, #818cf8)",
              fontFamily: "Syne, sans-serif",
            }}
          >
            {isLoading ? "Creating account..." : "Create account"}
          </button>
        </form>

        <div className="flex items-center gap-3 my-5">
          <div
            className="flex-1 h-px"
            style={{ background: "rgba(255,255,255,0.08)" }}
          />
          <span className="text-xs text-slate-600">or</span>
          <div
            className="flex-1 h-px"
            style={{ background: "rgba(255,255,255,0.08)" }}
          />
        </div>

        <p className="text-center text-sm text-slate-500">
          Already have an account?{" "}
          <Link
            to="/login"
            className="text-indigo-400 font-medium hover:underline"
          >
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}

export default RegisterPage;
