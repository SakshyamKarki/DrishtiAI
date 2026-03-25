import axios from "axios";
import { toast } from "react-toastify";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

export const TokenService = {
  getAccessToken: () => localStorage.getItem("access"),
  getRefreshToken: () => localStorage.getItem("refresh"),

  setTokens: (access, refresh) => {
    if (access) localStorage.setItem("access", access);
    if (refresh) localStorage.setItem("refresh", refresh);
  },

  removeTokens: () => {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
  },

  isTokenExpired: (token) => {
    if (!token) return true;
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      return payload.exp * 1000 < Date.now();
    } catch {
      return true;
    }
  },
};

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 3000,
  headers: {
    "Content-Type": "application/json",
  },
});

let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach((p) => (error ? p.reject(error) : p.resolve(error)));
  failedQueue = [];
};

const redirectToLogin = () => {
  if (window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
};

api.interceptors.request.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    if (!error.response) {
      toast.error("Network error. Please check your internet connection.");
      return Promise.reject(error);
    }

    const status = error.response?.status;
    const message = error.response?.data?.message;

    if (
      !originalRequest?.skipAuth &&
      status === 401 &&
      typeof error.response?.data?.detail === "string" &&
      error.response.data.detail.includes("token_invalid")
    ) {
      TokenService.removeTokens();
      toast.error("Session expired. Please login again.");
      redirectToLogin();
      return Promise.reject(error);
    }

    if (status === 401 && !originalRequest._retry) {
      if (originalRequest.url?.includes("/token/refresh/")) {
        TokenService.removeTokens();
        redirectToLogin();
        return Promise.reject(error);
      }
      originalRequest._retry = true;

      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({
            resolve: (token) => {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(api(originalRequest));
            },
            reject,
          });
        });
      }
      const refreshToken = TokenService.getRefreshToken();

      if (!refreshToken || TokenService.isTokenExpired(refreshToken)) {
        TokenService.removeTokens();
        redirectToLogin();
        return Promise.reject(error);
      }

      isRefreshing = true;
      try {
        // call refresh endpoint directly with plain axios, not our instance
        // so it doesnt go through this interceptor again
        const { data } = await axios.post(
          `${API_BASE_URL}/token/refresh/`,
          { refresh: refreshToken },
          { skipAuth: true },
        );

        // save new tokens
        TokenService.setTokens(data.access, data.refresh ?? refreshToken);

        // resolve all queued requests with the new token
        processQueue(null, data.access);

        // retry the original request that triggered the 401
        originalRequest.headers.Authorization = `Bearer ${data.access}`;
        return api(originalRequest);
      } catch (err) {
        processQueue(err, null);
        TokenService.removeTokens();
        redirectToLogin();
        return Promise.reject(err);
      } finally {
        isRefreshing = false;
      }
    }

    // ── All other errors ──
    const nonFieldErrors = error.response?.data?.error?.non_field_errors;
    const shouldShowToast =
      !originalRequest?.skipAuth &&
      !(status === 401 && originalRequest?._retry);

    if (shouldShowToast) {
      if (message) {
        toast.error(message);
      } else if (nonFieldErrors?.length) {
        toast.error(nonFieldErrors.join(", "));
      } else {
        // fallback messages by status code
        switch (status) {
          case 400:
            toast.error("Bad request. Please check your input.");
            break;
          case 403:
            toast.error("You are not authorized to do this.");
            break;
          case 404:
            toast.error("Requested resource not found.");
            break;
          case 422:
            toast.error("Validation failed. Please check your input.");
            break;
          case 500:
            toast.error("Server error. Please try again later.");
            break;
          default:
            toast.error("Something went wrong.");
            break;
        }
      }
    }

    return Promise.reject(error);
  },
);

// ─── Check Auth Status ────────────────────────────────────────────────────────
// call this on app load to verify if user is still authenticated
// used in your ProtectedRoute or app initialization
export const checkAuthStatus = async () => {
  try {
    const token = TokenService.getAccessToken();
    if (!token || TokenService.isTokenExpired(token)) return false;
    await api.get("/auth/me/", { skipAuth: true });
    return true;
  } catch {
    return false;
  }
};

export default api;
