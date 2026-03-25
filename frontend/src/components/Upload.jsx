import React, { useState } from "react";
import { predictImage } from "../api";

export default function Upload({ setResult }) {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    const res = await predictImage(file);
    setResult(res);
  };

  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
}