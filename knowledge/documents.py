"""Page-aware PDF and heading-aware Markdown chunking, no OCR."""
import hashlib
import json
from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_CONFIG = {'chunk_size': 1000, 'chunk_overlap': 150, 'loader_version': 1}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_chunks(path, root, source=None):
    path, root = Path(path), Path(root)
    source = source or {}
    filename = path.relative_to(root).as_posix()
    file_hash = digest(path.read_bytes())
    fingerprint = digest((file_hash + json.dumps(CHUNK_CONFIG, sort_keys=True) + json.dumps(source, sort_keys=True)).encode())
    sections = []
    if path.suffix.lower() == '.pdf':
        for number, page in enumerate(PdfReader(path).pages, 1):
            text = (page.extract_text() or '').replace('\x00', '').strip()
            if text:
                sections.append((text, number, ''))
        if not sections:
            raise ValueError(f'{filename} 无文本层，需要 OCR；本版本不支持扫描 PDF。')
    elif path.suffix.lower() == '.md':
        current, heading, heading_line = [], '文档开头', 1
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if re.match(r'^#{1,6}\s+', line):
                if current:
                    sections.append(('\n'.join(current), 0, f'{heading}（行{heading_line}）'))
                current, heading, heading_line = [], line.lstrip('#').strip(), line_number
            current.append(line)
        if current:
            sections.append(('\n'.join(current), 0, f'{heading}（行{heading_line}）'))
    else:
        raise ValueError(f'不支持的文档格式：{path.suffix}')
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,
                                              separators=['\n\n', '\n', '。', '. ', ' ', ''])
    chunks = []
    for text, page, heading in sections:
        for ordinal, chunk in enumerate(splitter.split_text(text)):
            if not chunk.strip():
                continue
            identifier = digest(f'{filename}:{fingerprint}:{page}:{heading}:{ordinal}'.encode())[:24]
            chunks.append(Document(page_content=chunk, metadata={
                'chunk_id': identifier, 'filename': filename,
                'title': source.get('title', path.stem), 'url': source.get('url', ''),
                'publication_date': source.get('publication_date', ''),
                'publisher': source.get('publisher', ''), 'page': page, 'heading': heading,
                'file_hash': file_hash,
            }))
    if not chunks:
        raise ValueError(f'{filename} 没有可入库文本。')
    return fingerprint, chunks


def collect_documents(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'资料目录不存在：{root}')
    manifest_path = root / 'sources.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    metadata = {entry['filename']: entry for entry in manifest}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.md'})
    if not paths:
        raise ValueError('没有 PDF 或 Markdown 文件，请先下载或添加资料。')
    return [(p.relative_to(root).as_posix(), *load_chunks(p, root, metadata.get(p.relative_to(root).as_posix()))) for p in paths]
