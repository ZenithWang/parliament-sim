from pathlib import Path
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from knowledge.documents import load_chunks, collect_documents


def test_markdown_headings_stable_ids_and_changes(tmp_path):
    path = tmp_path / 'notes.md'
    path.write_text('# First\n' + 'A long paragraph. ' * 150 + '\n## Second\nShort section.\n')
    fingerprint, chunks = load_chunks(path, tmp_path)
    assert all(len(d.page_content) <= 1000 for d in chunks)
    assert chunks[0].metadata['heading'] == 'First（行1）'
    assert chunks[-1].metadata['heading'].startswith('Second')
    assert chunks[-1].metadata['page'] == 0
    assert load_chunks(path, tmp_path) == (fingerprint, chunks)
    path.write_text('# First\nchanged')
    new_fingerprint, new_chunks = load_chunks(path, tmp_path)
    assert new_fingerprint != fingerprint
    assert new_chunks[0].metadata['chunk_id'] != chunks[0].metadata['chunk_id']


def test_pdf_physical_page_numbers_and_ocr(tmp_path):
    writer = PdfWriter()
    for text in ('First physical page', 'Second physical page'):
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                NameObject('/Subtype'): NameObject('/Type1'),
                                NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 20 200 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    path = tmp_path / 'text.pdf'
    writer.write(path)
    _, chunks = load_chunks(path, tmp_path, {'title': 'Test', 'url': 'https://example.org/report'})
    assert [d.metadata['page'] for d in chunks] == [1, 2]
    assert chunks[1].page_content == 'Second physical page'
    assert chunks[0].metadata['url'] == 'https://example.org/report'
    empty = PdfWriter()
    empty.add_blank_page(width=300, height=300)
    empty.write(tmp_path / 'scanned.pdf')
    with pytest.raises(ValueError, match='OCR'):
        load_chunks(tmp_path / 'scanned.pdf', tmp_path)


def test_empty_collection(tmp_path):
    with pytest.raises(ValueError, match='没有'):
        collect_documents(tmp_path)
