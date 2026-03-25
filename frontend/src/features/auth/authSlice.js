import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { TokenService } from "../../api/axiosInstance";
import { getMeApi, loginApi, logoutApi, registerApi } from "../../api/authApi";

export const loginThunk = createAsyncThunk(
    "auth/login",
    async (credentials, {rejectWithValue})=>{
        try{    
            const response = await loginApi(credentials);
            return response.data;
        }catch(error){
            return rejectWithValue(
                error.response?.data?.message || "Login Failed"
            );
        }
    }
);

export const registerThunk = createAsyncThunk(
    "auth/register",
    async (userData, {rejectWithValue})=>{
        try{
            const response = await registerApi(userData);
            return response.data;
        }catch(error){
            return rejectWithValue(
                error.response?.data?.message || "Registration Failed"
            );
        }
    }
);

export const logoutThunk = createAsyncThunk(
    "auth/logout",
    async (_, {rejectWithValue})=>{
        try{
            await logoutApi();
        }catch(error){
            //clear state
        }
    }
);

export const restoreSessionThunk = createAsyncThunk(
    "auth/restoreSession",
    async (_, {rejectWithValue})=>{
        try{
            const response = await getMeApi();
            return response.data;
        }catch(error){
            return rejectWithValue("Session expired.");
        }
    }
);


const initialState = {
    user: null,
    isAuthenticated: false,
    isLoading: false,
    error: null
};

const authSlice = createSlice({
    name: "auth",
    initialState,
    reducers:{
        clearError: (state)=>{
            state.error = null;
        },
    },
    extraReducers: (builder) => {
        builder
        .addCase(loginThunk.pending, (state)=>{
            state.isLoading = true;
            state.error = null;
        })
        .addCase(loginThunk.fulfilled, (state, action)=>{
            state.isLoading=false;
            state.isAuthenticated=true;
            state.user=action.payload.user;
            TokenService.setTokens(
                action.payload.access,
                action.payload.refresh
            );
        })
        .addCase(loginThunk.rejected, (state, action)=>{
            state.isLoading = false;
            state.isAuthenticated=false;
            state.error=action.payload;
        });
    }
});

export const {clearError} = authSlice.actions;

export const selectUser = (state) => state.auth.user;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;
export const selectAuthLoading = (state) => state.auth.isLoading;
export const selectAuthError = (state) => state.auth.error;

export default authSlice.reducer;