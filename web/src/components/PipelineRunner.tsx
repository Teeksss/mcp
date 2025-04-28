import React, { useState } from "react";
import { runPipeline } from "../api/mcp";

export default function PipelineRunner() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setLoading(true);
    const res = await runPipeline(query, { model_type: "gpt-4" });
    setResult(res);
    setLoading(false);
  }

  return (
    <div>
      <h2>Pipeline Çalıştır</h2>
      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Soru yazın..." />
      <button onClick={handleRun} disabled={loading || !query}>Çalıştır</button>
      {loading && <span>Çalışıyor...</span>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}