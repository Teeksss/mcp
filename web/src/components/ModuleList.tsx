import React, { useEffect, useState } from "react";
import { fetchModules } from "../api/mcp";

export default function ModuleList({ onSelect }: { onSelect: (mod: any) => void }) {
  const [modules, setModules] = useState<any[]>([]);
  useEffect(() => {
    fetchModules().then(setModules);
  }, []);

  return (
    <div>
      <h2>Modüller</h2>
      <ul>
        {modules.map((mod) => (
          <li key={mod.name}>
            <button onClick={() => onSelect(mod)}>{mod.name}</button>
          </li>
        ))}
      </ul>
    </div>
  );
}