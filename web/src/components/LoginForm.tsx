import React, { useState } from "react";
import { Form, Input, Button, message } from "antd";
import { useAuth } from "../context/AuthContext";
import { login } from "../api/auth";

export default function LoginForm() {
  const { login: setToken } = useAuth();
  const [loading, setLoading] = useState(false);
  const handleFinish = async (values: any) => {
    setLoading(true);
    try {
      const result = await login(values.username, values.password);
      setToken(result.access_token);
      message.success("Giriş başarılı!");
    } catch {
      message.error("Giriş başarısız!");
    }
    setLoading(false);
  };
  return (
    <Form onFinish={handleFinish}>
      <Form.Item name="username" rules={[{ required: true }]}>
        <Input placeholder="Kullanıcı adı" />
      </Form.Item>
      <Form.Item name="password" rules={[{ required: true }]}>
        <Input.Password placeholder="Şifre" />
      </Form.Item>
      <Button htmlType="submit" type="primary" block loading={loading}>
        Giriş Yap
      </Button>
    </Form>
  );
}