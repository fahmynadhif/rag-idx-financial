import os
import re
import json
import pdfplumber

# Folder input dan output
input_folder = "idx"
output_folder = "processed_text"
os.makedirs(output_folder, exist_ok=True)

def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)  # hilangkan newline ganda
    text = re.sub(r"\s{2,}", " ", text)     # hilangkan spasi ganda
    return text.strip()

def detect_and_insert_sections(text):
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if re.match(r"(?i)^laporan|^statement|^jumlah|^aset|^liabilitas", line.strip()):
            new_lines.append(f"\n## {line.strip()}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def extract_and_format_metadata(text, file_path=None, page_count=None):
    def extract(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_number(label):
        pattern = rf"{label}[\s:]*\(?([\d.,]+)\)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            pattern_alt = rf"{label}.*\n.*?\(?([\d.,]+)\)?"
            match = re.search(pattern_alt, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(".", "").replace(",", ".").strip()
            return f"-{value}" if "(" in match.group(0) and ")" in match.group(0) else value
        return ""

    def extract_subsidiaries(text):
        matches = re.findall(r"\bPT\s+([A-Z][a-zA-Z&() .,'-]+)\b", text)
        cleaned = []
        for m in matches:
            full_name = "PT " + m.strip()
            if not any(x in full_name.lower() for x in [
                "bertanggung jawab", "entitas", "ditandatangani",
                "informasi tertera", "entity name", "yang tidak memerlukan tanda tangan"
            ]):
                cleaned.append(full_name)
        unique_subs = sorted(set(cleaned))
        return ", ".join(unique_subs[:10])

    metadata = {
        "Kode Emiten": extract(r"Kode Emiten\s*:?[ \t]*([A-Z]+)"),
        "Nama Emiten": extract(r"Nama Emiten\s*:?[ \t]*(.+?)\n"),
        "Periode Laporan": extract(r"berakhir pada\s*(\d{2}/\d{2}/\d{4})"),
        "Jenis Laporan": extract(r"Laporan Keuangan\s+(.+?)(?:\n|$)"),
        "Mata Uang": extract(r"Mata uang pelaporan\s*:?[ \t]*(\w+)"),
        "Sektor": extract(r"Sektor\s*:?[ \t]*(.+?)\n"),
        "Subsektor": extract_subsidiaries(text),
        "Total Aset": extract_number("Jumlah aset"),
        "Laba Bersih": (
            extract_number("Laba bersih") or
            extract_number("Laba tahun berjalan") or
            extract_number("Laba") or
            extract_number("Current year profit") or
            extract_number("Profit for the year")
        ),
        "Liabilitas": extract_number("Jumlah liabilitas"),
        "Ekuitas": extract_number("Jumlah ekuitas"),
        "Pendapatan": (
            extract_number("Pendapatan bunga") or
            extract_number("Pendapatan")
        ),
        "Beban": (
            extract_number("Beban bunga") or
            extract_number("Beban operasional lainnya") or
            extract_number("Beban umum dan administrasi") or
            extract_number("Total beban") or
            extract_number("Total expenses") or
            extract_number("Beban")
        ),
        "Arus Kas Operasi": (
            extract_number("Kas dan setara kas akhir periode") or
            extract_number("Jumlah arus kas bersih ") or
            extract_number("Total net cash flows")
        )
    }

    if file_path and os.path.exists(file_path):
        metadata["Nama File"] = os.path.basename(file_path)
        metadata["Ukuran File"] = os.path.getsize(file_path)
    if page_count is not None:
        metadata["Jumlah Halaman"] = page_count

    meta_lines = [f"**{k}**: {v}" for k, v in metadata.items() if v]
    return metadata, "\n".join(meta_lines) + "\n\n"

def pdf_to_markdown(pdf_path, md_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_all_pages = []
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                text = clean_text(text)
                text = detect_and_insert_sections(text)
                text_all_pages.append(f"# Page {page_num + 1}\n\n{text}\n\n---\n")

        full_text = "\n".join(text_all_pages)
        metadata_dict, metadata_md = extract_and_format_metadata(full_text, pdf_path, len(pdf.pages))

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(metadata_md + full_text)

        json_path = md_path.replace(".md", ".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(metadata_dict, jf, indent=2, ensure_ascii=False)

        print(f"Berhasil mengekstrak dan menyimpan: {md_path} dan {json_path}")
    except Exception as e:
        print(f"Gagal memproses {pdf_path}: {e}")

def main():
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(output_folder, md_filename)
            pdf_to_markdown(pdf_path, md_path)

if __name__ == "__main__":
    main()
