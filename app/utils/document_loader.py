import os
from app.utils.pdf_parser import parse_pdf
from app.utils.text_parser import parse_text

def load_and_chunk_documents(data_dir):
    docs = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.endswith(".pdf"):
            docs.extend(parse_pdf(fpath))
        elif fname.endswith(".txt"):
            docs.extend(parse_text(fpath))
    return docs
