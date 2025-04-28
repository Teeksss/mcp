import React from "react";
import LlmPromptTester from "../components/LlmPromptTester";
import LiveDashboardWidgets from "../components/LiveDashboardWidgets";

export default function LlmPromptPage() {
  return (
    <div style={{ maxWidth: 1000, margin: "auto" }}>
      <LiveDashboardWidgets />
      <LlmPromptTester />
    </div>
  );
}