import {api} from './axiosInstance'

export const uploadDetectionImage = (file) => {
    const formData = new FormData();
    formData.append("image",file);

    return api.post('/detection/', formData, {
        headers : {
            "Content-Type":"multipart/form-data",
        }
    })
}