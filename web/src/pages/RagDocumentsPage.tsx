import React from "react";
import RagDocumentManager from "../components/RagDocumentManager";
import LiveDashboardWidgets from "../components/LiveDashboardWidgets";

export default function RagDocumentsPage() {
  return (
    <div style={{ maxWidth: 1000, margin: "auto" }}>
      <LiveDashboardWidgets />
      <RagDocumentManager />
    </div>
  );
}