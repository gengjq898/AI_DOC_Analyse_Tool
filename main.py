from fastapi import FastAPI, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os
from io import BytesIO
from docx import Document
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URL = os.getenv("SILICON_BASE_URL")
API_KEY = os.getenv("SILICON_API_KEY")
MODEL = os.getenv("MODEL_NAME")

# RAG全局变量
model_emb = SentenceTransformer('all-MiniLM-L6-v2')
index = None
chunk_list = []
chunk_size = 1000

# 文本分块
def split_text(text, size=1000):
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])
    return chunks

def read_docx(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])

@app.post("/upload")
async def upload(file: UploadFile):
    global index, chunk_list
    suffix = file.filename.split(".")[-1].lower()
    content = await file.read()
    if suffix == "txt":
        full_text = content.decode("utf-8", errors="ignore")
    elif suffix == "docx":
        full_text = read_docx(content)
    else:
        raise HTTPException(status_code=400, detail="仅支持 txt / docx")
    # debug
    print("文档前300字符：", full_text[:300])
    # 分块+向量化
    chunk_list = split_text(full_text, chunk_size)
    embeddings = model_emb.encode(chunk_list)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return {"msg": "✅上传成功，已加载文档（RAG分块完成）"}

@app.post("/chat")
async def chat(question: str = Body(..., embed=True)):
    global index, chunk_list
    if index is None or len(chunk_list)==0:
        return {"answer":"请先上传文档"}
    # 检索最相关2块
    q_emb = model_emb.encode([question])
    _, idx = index.search(q_emb, k=2)
    selected_chunks = "\n\n".join([chunk_list[i] for i in idx[0]])
    prompt = f"""基于下面文档片段回答用户问题。
没有相关信息就说文档无相关内容，禁止编造。

【文档片段】
{selected_chunks}

【用户问题】
{question}
"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    result = resp.json()
    answer = result["choices"][0]["message"]["content"]
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
