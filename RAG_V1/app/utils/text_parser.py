import os

def parse_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_text(text)
    filename = os.path.basename(path)
    return [
        {'text': chunk, 'metadata': {'source': filename, 'chunk': i, 'char_start': i*500, 'char_end': i*500+len(chunk)}}
        for i, chunk in enumerate(chunks)
    ]

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]