import pypdf

def parse_pdf(path):
    reader = pypdf.PdfReader(path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            for chunk in chunk_text(text):
                chunks.append({'text': chunk, 'metadata': {'page': i+1}})
    return chunks

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
