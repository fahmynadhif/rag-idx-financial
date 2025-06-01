import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_openai import ChatOpenAI

# Connect to Milvus
@st.cache_resource
def connect_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
        user="root",
        password="Milvus"
    )
    collection = Collection("financial_idx")
    collection.load()
    return collection

# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = ""  # ← Ganti dengan key kamu

# Load models
@st.cache_resource
def load_models():
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    collection = Collection("financial_idx")
    collection.load()
    results = collection.query(expr="", output_fields=["text"], limit=1000)
    corpus = [res["text"] for res in results if res.get("text")]

    if not corpus:
        corpus = ["laporan keuangan", "aset", "laba"]

    sparse_model = BM25SparseEmbedding(corpus=corpus)
    llm = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.7)
    return dense_model, sparse_model, llm

# Hybrid embed query
def embed_query(query, dense_model, sparse_model):
    dense_vec = dense_model.encode([query])[0]
    sparse_vec = sparse_model.embed_query(query)
    return dense_vec, sparse_vec

# Hybrid search
def hybrid_search(collection, dense_vec, sparse_vec, top_k=15):
    return collection.search(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=[
            "text", "file_name", "emiten_name", "emiten_code", "report_period", "report_type",
            "currency", "sector", "subsector",
            "total_assets", "net_profit", "liabilities", "equity",
            "revenue", "expenses", "operating_cash_flow"
        ]
    )

# Generate answer
def generate_answer(question, contexts, rag_model):
    context_text = "\n\n".join(contexts)
    prompt = f"Berikut ini adalah kutipan laporan keuangan:\n{context_text}\n\nPertanyaan: {question}\nJawaban:"
    response = rag_model.invoke(prompt)
    return response.content.strip()

# Streamlit UI
st.set_page_config(page_title="RAG IDX", page_icon="📊")
st.title("📊 RAG Tanya Jawab Laporan Keuangan IDX")

collection = connect_milvus()
dense_model, sparse_model, rag_model = load_models()

query = st.text_input("Tanyakan sesuatu tentang laporan keuangan:")

if query:
    with st.spinner("🔍 Mencari jawaban..."):
        dense_vec, sparse_vec = embed_query(query, dense_model, sparse_model)
        results = hybrid_search(collection, dense_vec, sparse_vec)

        if results[0]:
            # Ekstrak konteks sebagai list of strings
            contexts = [hit.entity.get("text", "-") for hit in results[0]]
            answer = generate_answer(query, contexts, rag_model)

            # ✅ Tampilkan jawaban
            st.subheader("🧠 Jawaban")
            st.write(answer)

            # ✅ Tampilkan sumber kutipan
            st.markdown("---")
            st.subheader("📚 Sumber Kutipan")

            for i, hit in enumerate(results[0]):
                ctx = hit.entity.get("text", "-")
                fname = hit.entity.get("file_name", "-")
                emiten = hit.entity.get("emiten_name", "-")
                period = hit.entity.get("report_period", "-")
                
                st.markdown(f"**📄 Chunk {i+1} dari {emiten} ({fname}) - {period}**")
                st.markdown(f"> {ctx.strip()}")
                st.markdown("---")
        else:
            st.warning("❌ Tidak ditemukan konteks relevan.")
