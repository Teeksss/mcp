import React, { useEffect, useRef, useState } from "react";
import { subscribeLogs } from "../api/mcp";

export default function LiveLogs() {
  const [logs, setLogs] = useState<string[]>([]);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sse = subscribeLogs((msg) => setLogs((prev) => [...prev, msg]));
    return () => sse.close();
  }, []);

  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div style={{ height: 220, overflowY: "auto", background: "#222", color: "#0f0", padding: 8 }}>
      {logs.slice(-100).map((l, i) => (
        <div key={i}>{l}</div>
      ))}
      <div ref={ref}></div>
    </div>
  );
}