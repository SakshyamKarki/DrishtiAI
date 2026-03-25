import React from "react";

export default function Result({ result }) {
  if (!result) return null;

  return (
    <div>
      <h2>Ensemble: {result.ensemble}</h2>
      <p>Confidence: {(result.confidence1 * 100).toFixed(2)}%</p>

      {result.resnet18 && (
        <>
          <h2>AI Detection: {result.resnet18}</h2>
          <p>Confidence: {(result.confidence2 * 100).toFixed(2)}%</p>
        </>
      )}
    </div>
  );
}