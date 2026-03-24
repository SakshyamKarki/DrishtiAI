import axios from "axios";

export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append("image", file);

  const res = await axios.post("http://127.0.0.1:8000/api/predict/", formData);

  return res.data;
};