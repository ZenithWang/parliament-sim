"""Opt in with RUN_REDIS_TESTS=1; only a UUID-named test namespace is modified."""
import os
from uuid import uuid4
import pytest
from langchain_core.embeddings import Embeddings
from knowledge.documents import collect_documents
from knowledge.store import KnowledgeStore, create_client

pytestmark = pytest.mark.skipif(os.getenv('RUN_REDIS_TESTS') != '1', reason='set RUN_REDIS_TESTS=1 for live Redis integration')


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0
        self.fail = False
    def embed_query(self, text):
        return [1.0, 0.5, 0.25]
    def embed_documents(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError('embedding failed')
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def store():
    client = create_client()
    name = 'parliament_sim_v2_test_' + uuid4().hex
    embeddings = FakeEmbeddings()
    instance = KnowledgeStore(embeddings, client, index_name=name, model='fixture')
    try:
        yield instance
    finally:
        client.execute_command('FT.DROPINDEX', name)
        for key in client.scan_iter(match=name + ':*'):
            client.delete(key)


def test_ingest_skip_replace_and_isolation(store, tmp_path):
    client = store.client
    before = set(client.execute_command('FT._LIST'))
    path = tmp_path / 'source.md'
    path.write_text('# Topic\nOriginal text.')
    files = collect_documents(tmp_path)
    assert store.ingest(files)[0]['status'] == 'updated'
    old_ids = store.manifest()['source.md']['ids']
    calls = store.embeddings.calls
    assert store.ingest(files)[0]['status'] == 'skipped'
    assert store.embeddings.calls == calls
    reopened = KnowledgeStore(store.embeddings, client, store.index_name, model='fixture', require_existing=True)
    hits = reopened.search('query')
    assert hits[0]['heading'] == 'Topic（行1）' and hits[0]['filename'] == 'source.md'
    path.write_text('# Updated\nNew content.')
    assert store.ingest(collect_documents(tmp_path))[0]['status'] == 'updated'
    assert not any(client.exists(store.prefix + ':' + i) for i in old_ids)
    assert 'New content' in store.search('query')[0]['content']
    assert set(client.execute_command('FT._LIST')) == before


def test_failed_replacement_preserves_old(store, tmp_path):
    path = tmp_path / 'source.md'
    path.write_text('Original')
    store.ingest(collect_documents(tmp_path))
    previous = store.manifest()
    path.write_text('Changed')
    store.embeddings.fail = True
    with pytest.raises(RuntimeError):
        store.ingest(collect_documents(tmp_path))
    assert store.manifest() == previous
    assert store.search('query')[0]['content'] == 'Original'


def test_configuration_mismatch(store):
    with pytest.raises(ValueError, match='不兼容'):
        KnowledgeStore(store.embeddings, store.client, store.index_name, model='different')


def test_empty_index_rejected(store):
    with pytest.raises(ValueError, match='为空'):
        KnowledgeStore(store.embeddings, store.client, store.index_name, model='fixture', require_existing=True)
