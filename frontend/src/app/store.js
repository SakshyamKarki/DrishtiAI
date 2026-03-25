import {configureStore} from '@reduxjs/toolkit'
import authReducer from '../features/auth/authSlice';
import historyReducer from '../features/history/historySlice';
import uploadReducer from '../features/upload/uploadSlice';

const store = configureStore({
    reducer:{
        auth: authReducer,
        upload: uploadReducer,
        history: historyReducer,
    },
});

export default store;
