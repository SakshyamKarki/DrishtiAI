import { useDispatch, useSelector } from "react-redux";
import { Link, useNavigate } from "react-router-dom";
import {
  loginThunk,
  selectAuthLoading,
  selectIsAuthenticated,
} from "../features/auth/authSlice";
import { useForm } from "react-hook-form";
import { yupResolver } from "@hookform/resolvers/yup";
import loginSchema from "../validations/loginSchema";
import { useEffect } from "react";
import '../styles/auth.css'

function LoginPage() {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const isLoading = useSelector(selectAuthLoading);
  const isAuthenticated = useSelector(selectIsAuthenticated);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(loginSchema),
  });

  useEffect(()=>{
    if (isAuthenticated) navigate("/");
  },[isAuthenticated]);

  const onSubmit = async (data) =>{
    const result = await dispatch(loginThunk(data));
    if(loginThunk.fulfilled.match(result)){
      navigate("/");
    }
  }
  return (
    <div className="auth-page">
      <div className="auth-orb-1"/>
      <div className="auth-orb-2"/>

      <div className="auth-grid"/>

      <div className="auth-card">
        <div className="auth-logo-row">
          <div className="auth-logo-ring">
            <div className="auth-logo-dot"/>
          </div>
          <span className="auth-logo-name">
            Drishti<span className="text-indigo-400">AI</span>
          </span>
        </div>
        <div className="auth-badge">
          <span className="auth-badge-dot" />
          Fake image detection
        </div>
        <h1
          className="text-2xl font-semibold text-slate-100 mb-1"
          style={{ fontFamily: "Syne, sans-serif" }}
        >
          Welcome back
        </h1>
        <p className="text-sm text-slate-500 mb-8">
          Sign in to your account to continue
        </p>
        <form onSubmit={handleSubmit(onSubmit)} noValidate>
          <div className="mb-5">
            <label
              htmlFor="email"
              className="block text-xs text-slate-400 mb-2 font-medium tracking-wide"
            >
              Email address
            </label>
            <input
              type="email"
              placeholder="you@example.com"
              {...register("email")}
              className="w-full rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-all"
              style={{
                background: errors.email
                  ? "rgba(248,113,113,0.05)"
                  : "rgba(255,255,255,0.05)",
                border: errors.email
                  ? "0.5px solid rgba(248,113,113,0.5)"
                  : "0.5px solid rgba(255,255,255,0.1)",
              }}
            />
            {errors.email && (
              <p className="text-xs text-red-400 mt-1.5">
                {errors.email.message}
              </p>
            )}
          </div>
          <div className="mb-2">
            <label className="block text-xs font-medium text-slate-400 mb-2 tracking-wide">
              Password
            </label>
            <input
              type="password"
              placeholder="••••••••"
              {...register("password")}
              className="w-full rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 outline-none transition-all"
              style={{
                background: errors.password
                  ? "rgba(248,113,113,0.05)"
                  : "rgba(255,255,255,0.05)",
                border: errors.password
                  ? "0.5px solid rgba(248,113,113,0.5)"
                  : "0.5px solid rgba(255,255,255,0.1)",
              }}
            />
            {errors.password && (
              <p className="text-xs text-red-400 mt-1.5">
                {errors.password.message}
              </p>
            )}
          </div>
          <div className="text-right mb-6">
            {/* <span className="text-xs text-indigo-400 cursor-pointer hover:underline">
              Forgot password?
            </span> */}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-xl text-sm font-semibold text-white transition-all disabled:opacity-60"
            style={{
              background: "linear-gradient(135deg, #6366f1, #818cf8)",
              fontFamily: "Syne, sans-serif",
            }}
          >
            {isLoading ? "Signing in..." : "Sign in"}
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
          Don't have an account?{" "}
          <Link
            to="/register"
            className="text-indigo-400 font-medium hover:underline"
          >
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
