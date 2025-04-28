import React from "react";
import { Menu } from "antd";
import { CloudDownloadOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { Link, useLocation } from "react-router-dom";

const SidebarMenu = () => {
  const location = useLocation();
  return (
    <Menu
      theme="dark"
      mode="inline"
      selectedKeys={[location.pathname]}
      style={{ height: "100%", borderRight: 0 }}
    >
      <Menu.Item key="/rag-documents" icon={<CloudDownloadOutlined />}>
        <Link to="/rag-documents">RAG Dökümanları</Link>
      </Menu.Item>
      <Menu.Item key="/llm-prompt" icon={<ThunderboltOutlined />}>
        <Link to="/llm-prompt">LLM Prompt Test</Link>
      </Menu.Item>
    </Menu>
  );
};

export default SidebarMenu;