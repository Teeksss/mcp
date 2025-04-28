import React from "react";
import { Select } from "antd";
import { useAuth } from "../context/AuthContext";

export default function RoleSwitcher() {
  const { role, setRole } = useAuth();
  return (
    <div style={{ float: "right", marginRight: 16 }}>
      <Select value={role} onChange={setRole} style={{ width: 120 }}>
        <Select.Option value="viewer">Seyirci</Select.Option>
        <Select.Option value="admin">Yönetici</Select.Option>
      </Select>
    </div>
  );
}