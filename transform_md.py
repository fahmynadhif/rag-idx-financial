import os
import re
import json
import pdfplumber
from pymilvus import Collection, connections, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

input_folder = "idx"
output_folder = "processed_text"
os.makedirs(output_folder, exist_ok=True)

def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def extract_and_format_metadata(text, file_path=None, page_count=None):
    def extract(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_number(label):
        pattern = rf"{label}[:\s]*\(?([\d.,]+)\)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            pattern_alt = rf"{label}.*\n.*?\(?([\d.,]+)\)?"
            match = re.search(pattern_alt, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            try:
                cleaned = re.sub(r"[^\d.,]", "", raw)
                if cleaned.count(",") == 1 and cleaned.count(".") > 1:
                    cleaned = cleaned.replace(".", "").replace(",", ".")
                else:
                    cleaned = cleaned.replace(",", "").replace(".", ".")
                return f"-{cleaned}" if "(" in match.group(0) and ")" in match.group(0) else cleaned
            except ValueError:
                return "0"
        return ""

    def extract_subsidiaries(text):
        matches = re.findall(r"\bPT\s+([A-Z][a-zA-Z&() .,'-]+)\b", text)
        cleaned = []
        for m in matches:
            full_name = "PT " + m.strip()
            if not any(x in full_name.lower() for x in ["bertanggung jawab", "entitas", "ditandatangani", "informasi tertera", "entity name", "yang tidak memerlukan tanda tangan"]):
                cleaned.append(full_name)
        joined = ", ".join(sorted(set(cleaned))[:10])
        return joined[:250]  # truncate to max length allowed by Milvus field

    def safe_float(value):
        try:
            return float(value)
        except ValueError:
            return 0.0

    metadata = {
        "file_name": os.path.basename(file_path),
        "filesize": os.path.getsize(file_path) if file_path else 0,
        "emiten_code": extract(r"Kode Emiten\s*:?[ \t]*([A-Z]+)"),
        "emiten_name": extract(r"Nama Emiten\s*:?[ \t]*(.+?)\n"),
        "report_period": extract(r"berakhir pada\s*(\d{2}/\d{2}/\d{4})"),
        "report_type": extract(r"Laporan Keuangan\s+(.+?)(?:\n|$)"),
        "currency": extract(r"Mata uang pelaporan\s*:?[ \t]*(\w+)"),
        "sector": extract(r"Sektor\s*:?[ \t]*(.+?)\n"),
        "subsector": extract_subsidiaries(text),
        "total_assets": safe_float(extract_number("Jumlah aset") or "0"),
        "net_profit": safe_float(
            extract_number("Laba bersih") or
            extract_number("Laba tahun berjalan") or
            extract_number("Profit for the year") or "0"),
        "liabilities": safe_float(extract_number("Jumlah liabilitas") or "0"),
        "equity": safe_float(extract_number("Jumlah ekuitas") or "0"),
        "revenue": safe_float(
            extract_number("Pendapatan bunga") or
            extract_number("Pendapatan") or "0"),
        "expenses": safe_float(
            extract_number("Beban bunga") or
            extract_number("Beban umum dan administrasi") or
            extract_number("Total beban") or
            extract_number("Total expenses") or "0"),
        "operating_cash_flow": safe_float(
            extract_number("Kas dan setara kas akhir periode") or
            extract_number("Jumlah arus kas bersih") or
            extract_number("Total net cash flows") or "0"),
        "pages": page_count or 0
    }
    return metadata

def chunk_text_with_metadata(text, metadata):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text], metadatas=[metadata])

def connect_to_milvus():
    connections.connect(
        alias="default",
        host="localhost",
        port="19530",
        user="root",
        password="Milvus"
    )

def create_collection():
    schema = CollectionSchema([
        FieldSchema("doc_id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=384),
        FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("file_name", DataType.VARCHAR, max_length=256),
        FieldSchema("emiten_name", DataType.VARCHAR, max_length=128),
        FieldSchema("emiten_code", DataType.VARCHAR, max_length=16),
        FieldSchema("report_period", D ataType.VARCHAR, max_length=32),
        FieldSchema("report_type", DataType.VARCHAR, max_length=128),
        FieldSchema("currency", DataType.VARCHAR, max_length=8),
        FieldSchema("sector", DataType.VARCHAR, max_length=64),
        FieldSchema("subsector", DataType.VARCHAR, max_length=256),
        FieldSchema("total_assets", DataType.DOUBLE),
        FieldSchema("net_profit", DataType.DOUBLE),
        FieldSchema("liabilities", DataType.DOUBLE),
        FieldSchema("equity", DataType.DOUBLE),
        FieldSchema("revenue", DataType.DOUBLE),
        FieldSchema("expenses", DataType.DOUBLE),
        FieldSchema("operating_cash_flow", DataType.DOUBLE),
        FieldSchema("pages", DataType.INT64),
    ])
    collection = Collection("financial_idx", schema)
    collection.create_index("dense_vector", {"index_type": "FLAT", "metric_type": "IP"})
    collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
    collection.load()
    return collection

def insert_data(collection, documents, dense_vectors, sparse_vectors):
    texts = [doc.page_content for doc in documents]
    metas = [doc.metadata for doc in documents]
    data = [
        dense_vectors,
        sparse_vectors,
        texts,
        [m['file_name'] for m in metas],
        [m['emiten_name'] for m in metas],
        [m['emiten_code'] for m in metas],
        [m['report_period'] for m in metas],
        [m['report_type'] for m in metas],
        [m['currency'] for m in metas],
        [m['sector'] for m in metas],
        [m['subsector'] for m in metas],
        [m['total_assets'] for m in metas],
        [m['net_profit'] for m in metas],
        [m['liabilities'] for m in metas],
        [m['equity'] for m in metas],
        [m['revenue'] for m in metas],
        [m['expenses'] for m in metas],
        [m['operating_cash_flow'] for m in metas],
        [m['pages'] for m in metas],
    ]
    collection.insert(data)
def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text_all = []
        for i, page in enumerate(pdf.pages):
            text = clean_text(page.extract_text() or "")
            text_all.append(text)
        full_text = "\n".join(text_all)

    metadata = extract_and_format_metadata(full_text, pdf_path, len(text_all))
    documents = chunk_text_with_metadata(full_text, metadata)

    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dense_vectors = dense_model.encode([doc.page_content for doc in documents], show_progress_bar=True)
    sparse_model = BM25SparseEmbedding(corpus=[doc.page_content for doc in documents])
    sparse_vectors = sparse_model.embed_documents([doc.page_content for doc in documents])

    return documents, dense_vectors, sparse_vectors

def main():
    connect_to_milvus()
    collection = create_collection()

    for file in os.listdir(input_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            docs, dense_vecs, sparse_vecs = process_pdf(pdf_path)
            insert_data(collection, docs, dense_vecs, sparse_vecs)
            print(f"✔️ Inserted {file} into Milvus")

if __name__ == "__main__":
    main()
