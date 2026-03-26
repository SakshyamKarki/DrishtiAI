import api from "./axiosInstance";

export const loginApi = (credentials) => api.post("/auth/login/",credentials);

export const registerApi = (userData) => api.post("/auth/register/", userData);

export const logoutApi = () => api.post("/auth/logout/", {
    refresh: localStorage.getItem('refresh')
})

export const getMeApi = () => api.get("/auth/me/");