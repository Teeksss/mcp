import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "antd";
import SidebarMenu from "./SidebarMenu";
import RagDocumentsPage from "./pages/RagDocumentsPage";
import LlmPromptPage from "./pages/LlmPromptPage";

const { Content, Sider } = Layout;

function App() {
  return (
    <BrowserRouter>
      <Layout style={{ minHeight: "100vh" }}>
        <Sider>
          <SidebarMenu />
        </Sider>
        <Layout>
          <Content style={{ margin: 24 }}>
            <Routes>
              <Route path="/rag-documents" element={<RagDocumentsPage />} />
              <Route path="/llm-prompt" element={<LlmPromptPage />} />
              <Route path="*" element={<RagDocumentsPage />} />
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </BrowserRouter>
  );
}

export default App;