import glob, asyncio
from src.services.rag_service import RAGService

rag = RAGService()

async def bulk_import_txt(folder):
    files = glob.glob(f"{folder}/*.txt")
    for path in files:
        with open(path, encoding="utf8") as f:
            content = f.read()
        await rag.add_document({"content": content, "metadata": {"filename": path}})

asyncio.run(bulk_import_txt("data_folder"))