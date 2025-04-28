import React from "react";
import { Modal, Form, Input, Select, Button, message } from "antd";
import { createModule, updateModule } from "../api/mcp";

export default function ModuleForm({ initial, onSuccess, onCancel }: any) {
  const [form] = Form.useForm();
  const [visible, setVisible] = React.useState(!!initial);

  React.useEffect(() => {
    if (initial) {
      form.setFieldsValue(initial);
      setVisible(true);
    }
  }, [initial]);

  function handleFinish(values: any) {
    const apiCall = initial ? updateModule(initial.id, values) : createModule(values);
    apiCall.then(() => {
      message.success(initial ? "Güncellendi" : "Eklendi");
      setVisible(false);
      form.resetFields();
      onSuccess?.();
    });
  }

  return (
    <>
      {!initial && (
        <Button type="primary" onClick={() => setVisible(true)} style={{ marginBottom: 12 }}>
          Modül Ekle
        </Button>
      )}
      <Modal
        open={visible}
        title={initial ? "Modül Düzenle" : "Modül Ekle"}
        onCancel={() => { setVisible(false); onCancel?.(); }}
        onOk={() => form.submit()}
        okText={initial ? "Kaydet" : "Ekle"}
      >
        <Form form={form} layout="vertical" onFinish={handleFinish}>
          <Form.Item name="name" label="Modül Adı" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="type" label="Tip" rules={[{ required: true }]}>
            <Select options={[
              { label: "Model", value: "model" },
              { label: "Veri Kaynağı", value: "data" },
              { label: "RAG", value: "rag" },
            ]} />
          </Form.Item>
          <Form.Item name="status" label="Durum" rules={[{ required: true }]}>
            <Select options={[
              { label: "Aktif", value: "active" },
              { label: "Pasif", value: "inactive" },
            ]} />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}