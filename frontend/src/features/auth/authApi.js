import api from "../../api/axiosInstance";

export const loginApi = (credentials) => api.post("/auth/login/",credentials);

export const registerApi = (userData) => api.post("/auth/register/", userData);

// export const logoutApi = () => 

// export const getMeApi = () => {}