import React from "react";

export default function ModuleDetail({ mod }: { mod: any }) {
  if (!mod) return <div>Bir modül seçin.</div>;
  return (
    <div>
      <h3>{mod.name}</h3>
      <pre>{JSON.stringify(mod, null, 2)}</pre>
    </div>
  );
}