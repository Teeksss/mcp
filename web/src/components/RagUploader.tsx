import { Upload, Button, message } from "antd";
import { UploadOutlined } from "@ant-design/icons";

export default function RagUploader() {
  const props = {
    accept: ".txt,.pdf",
    customRequest: async ({file, onSuccess}) => {
      const text = await file.text();
      const res = await fetch("/api/v1/rag/documents/add", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({content: text, metadata: {filename: file.name}})
      });
      if (res.ok) {
        message.success("Yüklendi!");
        onSuccess();
      } else {
        message.error("Yüklenemedi!");
      }
    }
  };
  return (
    <Upload {...props} showUploadList={false}>
      <Button icon={<UploadOutlined />}>Döküman Yükle</Button>
    </Upload>
  );
}