import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { TokenService } from "../../api/axiosInstance";

// export const loginThunk = createAsyncThunk()

// export const registerThunk = createAsyncThunk()

// export const logoutThunk = createAsyncThunk()

// export const restoreSessionThunk = createAsyncThunk()


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
        loginStart: (state)=>{
            state.isLoading= true;
            state.error = null;
        },
        loginSuccess: (state, action)=>{
            state.isLoading = false;
            state.isAuthenticated = true;
            state.user = action.payload.user;
            TokenService.setTokens(
                action.payload.access,
                action.payload.refresh
            );
        },
        loginFailure: (state, action)=>{
            state.isLoading = false;
            state.isAuthenticated = false;
            state.error = action.payload;
            state.user = null;
        },
        logout: (state)=>{
            state.user = null;
            state.isAuthenticated = false;
            state.isLoading = false;
            state.error = null;
            TokenService.removeTokens(); 
        },
        restoreSession: (state, action)=>{
            state.user = action.payload.user;
            state.isAuthenticated = true;
        },
        clearError: (state)=>{
            state.error = null;
        },
    },
    extraReducers: (builder) => {

    }
});

export const {clearError, loginStart} = authSlice.actions;

export const selectUser = (state) => state.auth.user;
export const selectIsAuthenticated = (state) => state.auth.isAuthenticated;
export const selectAuthLoading = (state) => state.auth.isLoading;
export const selectAuthError = (state) => state.auth.error;

export default authSlice.reducer;