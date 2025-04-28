import React, { useState, useEffect } from "react";
import {
  Upload,
  Button,
  message,
  List,
  Input,
  Modal,
  Space,
  Typography,
  Card,
  Select,
  Form,
  Tabs
} from "antd";
import {
  UploadOutlined,
  CloudDownloadOutlined,
  PlusOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  LinkOutlined
} from "@ant-design/icons";
import Tesseract from "tesseract.js";
import "pdfjs-dist/build/pdf.worker.entry";
import pdfjsLib from "pdfjs-dist/build/pdf";

const { TextArea } = Input;
const { Title } = Typography;

type RagDocument = {
  id?: string;
  content: string;
  metadata: Record<string, any>;
};

function isPDF(file: File) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}
function isImage(file: File) {
  return file.type.startsWith("image/") && !isPDF(file);
}

async function pdfToPageTexts(file: File): Promise<{ text: string; page: number }[]> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pages: { text: string; page: number }[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((item: any) => item.str).join(" ") + "\n";
    pages.push({ text: pageText.trim(), page: i });
  }
  return pages;
}

async function ocrImage(file: File, langs = "eng+tur"): Promise<string> {
  const worker = await Tesseract.createWorker(langs);
  const { data } = await worker.recognize(file);
  await worker.terminate();
  return data.text;
}

async function fetchWebText(url: string): Promise<string> {
  // Backend proxy gerektirir
  const resp = await fetch(`/api/v1/rag/fetch_web?url=${encodeURIComponent(url)}`);
  const data = await resp.json();
  return data.text || "";
}

export default function RagDocumentManager() {
  const [documents, setDocuments] = useState<RagDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newDocContent, setNewDocContent] = useState("");
  const [newDocMeta, setNewDocMeta] = useState("{}");
  const [backend, setBackend] = useState("chromadb");
  const [refresh, setRefresh] = useState(0);
  const [ocrLang, setOcrLang] = useState("eng+tur");
  const [tab, setTab] = useState("manual");
  const [webUrl, setWebUrl] = useState("");
  const [webLoading, setWebLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/v1/rag/documents/list?backend=${backend}`)
      .then(r => r.json())
      .then(docs => setDocuments(Array.isArray(docs) ? docs : []))
      .catch(() => setDocuments([]))
      .finally(() => setLoading(false));
  }, [refresh, backend]);

  const handleAddDocument = async () => {
    try {
      const meta = JSON.parse(newDocMeta || "{}");
      const resp = await fetch(`/api/v1/rag/documents/add?backend=${backend}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newDocContent, metadata: meta })
      });
      if (resp.ok) {
        message.success("Döküman eklendi!");
        setShowAddModal(false);
        setRefresh(r => r + 1);
        setNewDocContent("");
        setNewDocMeta("{}");
      } else {
        message.error("Ekleme başarısız!");
      }
    } catch (e) {
      message.error("Geçersiz metadata JSON!");
    }
  };

  const uploadProps = {
    accept: ".txt,.pdf,.png,.jpg,.jpeg,.bmp,.tiff,.webp",
    showUploadList: false,
    customRequest: async ({ file, onSuccess }: any) => {
      let meta: any = { filename: file.name, type: file.type };
      try {
        if (isPDF(file)) {
          message.loading({ content: "PDF sayfalar bölünüyor...", key: "pdf" });
          const pages = await pdfToPageTexts(file);
          let successCount = 0;
          for (const { text, page } of pages) {
            if (text.trim() === "") continue;
            const resp = await fetch(`/api/v1/rag/documents/add?backend=${backend}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                content: text,
                metadata: { ...meta, page }
              })
            });
            if (resp.ok) successCount++;
          }
          setRefresh(r => r + 1);
          message.success({ content: `PDF bölündü, ${successCount} sayfa eklendi!`, key: "pdf" });
          onSuccess();
          return;
        } else if (isImage(file)) {
          message.loading({ content: `OCR (${ocrLang}) yapılıyor...`, key: "ocr" });
          const text = await ocrImage(file, ocrLang);
          if (!text.trim()) throw new Error("OCR sonucu boş");
          const resp = await fetch(`/api/v1/rag/documents/add?backend=${backend}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: text, metadata: meta })
          });
          if (resp.ok) {
            setRefresh(r => r + 1);
            message.success({ content: "Resimden metin çıkarıldı ve kaydedildi!", key: "ocr" });
            onSuccess();
          } else {
            message.error("Yüklenemedi!");
          }
          return;
        } else {
          const text = await file.text();
          const resp = await fetch(`/api/v1/rag/documents/add?backend=${backend}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: text, metadata: meta })
          });
          if (resp.ok) {
            setRefresh(r => r + 1);
            message.success("Döküman yüklendi!");
            onSuccess();
          } else {
            message.error("Yüklenemedi!");
          }
        }
      } catch (e) {
        message.error("Dosya işlenemedi: " + (e as any).message);
      }
    }
  };

  const handleFetchWeb = async () => {
    if (!/^https?:\/\//.test(webUrl)) {
      message.error("Geçerli bir web adresi girin!");
      return;
    }
    setWebLoading(true);
    try {
      const text = await fetchWebText(webUrl);
      if (!text || text.length < 20) {
        message.error("Web kaynağından anlamlı metin çıkarılamadı.");
        setWebLoading(false);
        return;
      }
      const resp = await fetch(`/api/v1/rag/documents/add?backend=${backend}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: text, metadata: { source: webUrl } })
      });
      if (resp.ok) {
        message.success("Web kaynağı eklendi!");
        setRefresh(r => r + 1);
        setWebUrl("");
      } else {
        message.error("Web kaynağı eklenemedi!");
      }
    } catch (e) {
      message.error("Web çekme hatası: " + (e as any).message);
    }
    setWebLoading(false);
  };

  const handleBackendChange = (val: string) => {
    setBackend(val);
    setRefresh(r => r + 1);
  };

  return (
    <Card
      title={<Space><CloudDownloadOutlined /> RAG Döküman Yükle/Yönet</Space>}
      extra={
        <Space>
          <Select value={backend} onChange={handleBackendChange} style={{ width: 140 }}>
            <Select.Option value="chromadb">ChromaDB</Select.Option>
            <Select.Option value="faiss">FAISS</Select.Option>
            <Select.Option value="pinecone">Pinecone</Select.Option>
            <Select.Option value="elasticsearch">ElasticSearch</Select.Option>
          </Select>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setShowAddModal(true)}>
            Döküman Ekle
          </Button>
          <Upload {...uploadProps}>
            <Button icon={<UploadOutlined />}>Dosya Yükle</Button>
          </Upload>
        </Space>
      }
      style={{ margin: 32 }}
    >
      <List
        loading={loading}
        bordered
        dataSource={documents}
        rowKey={doc => doc.id || doc.content.slice(0, 32)}
        renderItem={doc => (
          <List.Item>
            <Space direction="vertical" style={{ width: "100%" }}>
              <b>
                {(doc.metadata?.filename && doc.metadata.filename.endsWith(".pdf")) && <FilePdfOutlined style={{ color: "#e53935" }} />}
                {(doc.metadata?.filename && /\.(png|jpg|jpeg|bmp|tiff|webp)$/i.test(doc.metadata.filename)) && <FileImageOutlined style={{ color: "#1890ff" }} />}
                {doc.metadata?.filename || doc.metadata?.source || "Döküman"}
              </b>
              <div style={{ whiteSpace: "pre-wrap", maxHeight: 120, overflow: "auto", fontSize: 13 }}>
                {doc.content.slice(0, 400)}{doc.content.length > 400 ? "..." : ""}
              </div>
              {doc.metadata && (
                <pre style={{ background: "#fafafa", fontSize: 12, margin: 0 }}>
                  {JSON.stringify(doc.metadata, null, 2)}
                </pre>
              )}
            </Space>
          </List.Item>
        )}
      />
      <Modal
        title={
          <Tabs activeKey={tab} onChange={setTab} items={[
            { key: "manual", label: "Metin", children: null },
            { key: "web", label: <><LinkOutlined /> Web</>, children: null }
          ]} />
        }
        open={showAddModal}
        onOk={tab === "manual" ? handleAddDocument : handleFetchWeb}
        onCancel={() => setShowAddModal(false)}
        okText={tab === "manual" ? "Ekle" : "Web'den Çek"}
        cancelText="Vazgeç"
        footer={null}
      >
        <Tabs
          activeKey={tab}
          onChange={setTab}
          items={[
            {
              key: "manual", label: "Metin",
              children: (
                <Form
                  onFinish={handleAddDocument}
                  layout="vertical"
                >
                  <Form.Item label="Döküman içeriği" required>
                    <TextArea
                      rows={5}
                      placeholder="Döküman içeriği"
                      value={newDocContent}
                      onChange={e => setNewDocContent(e.target.value)}
                      style={{ marginBottom: 12 }}
                    />
                  </Form.Item>
                  <Form.Item label='Metadata (örn: {"source":"manual"})'>
                    <Input
                      value={newDocMeta}
                      onChange={e => setNewDocMeta(e.target.value)}
                    />
                  </Form.Item>
                  <Form.Item>
                    <Button type="primary" htmlType="submit">Ekle</Button>
                  </Form.Item>
                </Form>
              )
            },
            {
              key: "web", label: <><LinkOutlined /> Web</>,
              children: (
                <Form
                  onFinish={handleFetchWeb}
                  layout="vertical"
                >
                  <Form.Item label="Web Sayfası URL" required>
                    <Input
                      placeholder="https://..."
                      value={webUrl}
                      onChange={e => setWebUrl(e.target.value)}
                    />
                  </Form.Item>
                  <Form.Item>
                    <Button type="primary" htmlType="submit" loading={webLoading}>
                      Web'den Çek ve Ekle
                    </Button>
                  </Form.Item>
                </Form>
              )
            }
          ]}
        />
        <div style={{ marginTop: 16 }}>
          <b>OCR Dil Seçimi (resimler için):</b>
          <Select value={ocrLang} style={{ width: 180, marginLeft: 8 }} onChange={setOcrLang}>
            <Select.Option value="eng+tur">Türkçe + İngilizce</Select.Option>
            <Select.Option value="eng">İngilizce</Select.Option>
            <Select.Option value="tur">Türkçe</Select.Option>
            <Select.Option value="deu">Almanca</Select.Option>
            <Select.Option value="fra">Fransızca</Select.Option>
          </Select>
        </div>
      </Modal>
    </Card>
  );
}