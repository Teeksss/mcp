import React, { useEffect, useState } from "react";
import { fetchHealth } from "../api/mcp";

export default function HealthStatus() {
  const [status, setStatus] = useState<string>("...");
  useEffect(() => {
    fetchHealth().then((d) => setStatus(d.status));
  }, []);
  return (
    <div>
      <span>Sunucu Sağlık: </span>
      <b style={{ color: status === "ok" ? "green" : "red" }}>{status}</b>
    </div>
  );
}