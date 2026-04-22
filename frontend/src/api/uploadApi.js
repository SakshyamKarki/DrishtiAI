import { api } from './axiosInstance'

export const uploadDetectionImage = (file) => {
    const formData = new FormData();
    formData.append("image", file);

    return api.post('/detection/', formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
        timeout: 120000, // 2 min timeout for heavy inference
    });
};

export const getDetectionHistory = (limit = 50) => {
    return api.get(`/detection/?limit=${limit}`);
};

export const getDetectionStats = () => {
    return api.get('/detection/stats/');
};
