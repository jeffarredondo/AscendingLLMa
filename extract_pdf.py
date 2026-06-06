from pypdf import PdfReader

def extract_pdf_text(pdf_path, output_path):
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"  {total_pages} pages")

    text = []
    for i, page in enumerate(reader.pages):
        if i % 50 == 0:
            print(f"  Extracting page {i}/{total_pages}...")
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)

    full_text = "\n".join(text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"  Extracted {len(full_text):,} characters")
    print(f"  Saved to {output_path}")
    return full_text

if __name__ == "__main__":
    extract_pdf_text("spacex_s1.pdf", "spacex_s1.txt")