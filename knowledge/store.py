"""Redis vector store with explicit configuration and per-file replacement."""
import json
import os

from langchain_redis import RedisConfig, RedisVectorStore
from redis import Redis

from config import load_config
from knowledge.documents import CHUNK_CONFIG

INDEX = 'parliament_sim_v2'
EMBEDDING_MODEL = 'text-embedding-v3'


def create_embeddings():
    load_config()
    if not os.getenv('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('缺少 DASHSCOPE_API_KEY，请在项目 .env 或环境变量中配置。')
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings(model=EMBEDDING_MODEL, max_retries=1,
                               dashscope_api_key=os.environ['DASHSCOPE_API_KEY'])


def create_client():
    load_config()
    client = Redis.from_url(os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0'),
                           socket_connect_timeout=5, socket_timeout=10)
    client.ping()
    info = client.execute_command('COMMAND', 'INFO', 'FT.SEARCH')
    if not info or not any(info.values() if isinstance(info, dict) else info):
        raise ValueError('Redis 未启用 Search 向量检索能力。')
    return client


class KnowledgeStore:
    def __init__(self, embeddings, client, index_name=INDEX, model=EMBEDDING_MODEL,
                 require_existing=False):
        if not (index_name == INDEX or index_name.startswith(INDEX + '_test_')):
            raise ValueError('只允许本项目索引或隔离测试索引。')
        self.client, self.embeddings, self.index_name = client, embeddings, index_name
        self.prefix = index_name + ':doc'
        self.meta_key = index_name + ':manifest'
        self.config_key = index_name + ':config'
        existing_indexes = {v.decode() if isinstance(v, bytes) else v for v in client.execute_command('FT._LIST')}
        raw = client.get(self.config_key)
        if require_existing and (not raw or index_name not in existing_indexes):
            raise ValueError('知识库索引不存在，请先运行 python -m knowledge.ingest。')
        if not raw and index_name in existing_indexes:
            raise ValueError('索引缺少本项目配置记录，请显式重建项目索引。')
        if raw and index_name not in existing_indexes:
            raise ValueError('索引与配置不一致，请显式重建项目索引。')
        dimensions = len(embeddings.embed_query('dimension compatibility check'))
        self.settings = {'model': model, 'dimensions': dimensions, **CHUNK_CONFIG}
        if raw and json.loads(raw) != self.settings:
            raise ValueError('嵌入模型、维度或切分配置不兼容；请用 ingest --rebuild 重建本项目索引。')
        config = RedisConfig(index_name=index_name, key_prefix=self.prefix,
                             redis_client=client, embedding_dimensions=dimensions,
                             distance_metric='COSINE', indexing_algorithm='FLAT',
                             legacy_key_format=False,
                             metadata_schema=[{'name': 'chunk_id', 'type': 'tag'},
                                              {'name': 'filename', 'type': 'tag'},
                                              {'name': 'page', 'type': 'numeric'}])
        self.vector_store = RedisVectorStore(embeddings, config=config)
        if not raw:
            client.set(self.config_key, json.dumps(self.settings, sort_keys=True))
        if require_existing and not self.manifest():
            raise ValueError('知识库为空，请先运行 python -m knowledge.ingest。')
        if require_existing:
            result = client.execute_command('FT.SEARCH', index_name, '*', 'LIMIT', 0, 0)
            count = result.get(b'total_results', result.get('total_results', 0)) if isinstance(result, dict) else result[0]
            if count == 0:
                raise ValueError('知识库索引没有文档，请重新入库。')

    @classmethod
    def open_for_search(cls):
        client = create_client()
        return cls(create_embeddings(), client, require_existing=True)

    def manifest(self):
        raw = self.client.get(self.meta_key)
        return json.loads(raw) if raw else {}

    def ingest(self, files):
        results = []
        with self.client.lock(self.index_name + ':ingest_lock', timeout=1800, blocking_timeout=5):
            manifest = self.manifest()
            for filename, fingerprint, chunks in files:
                previous = manifest.get(filename)
                if previous and previous['fingerprint'] == fingerprint:
                    results.append({'filename': filename, 'status': 'skipped', 'chunks': len(previous['ids'])})
                    continue
                identifiers = [d.metadata['chunk_id'] for d in chunks]
                try:
                    # DashScope accepts at most 25 inputs per request. Batch here,
                    # because vector_store.add_texts embeds its entire input first.
                    for offset in range(0, len(chunks), 20):
                        batch = chunks[offset:offset + 20]
                        self.vector_store.add_texts([d.page_content for d in batch],
                            metadatas=[d.metadata for d in batch], keys=[d.metadata['chunk_id'] for d in batch])
                except Exception:
                    old_ids = set(previous['ids']) if previous else set()
                    self.vector_store.delete(ids=[i for i in identifiers if i not in old_ids])
                    raise
                # Only after successful writes, atomically retire this file's old
                # keys and publish its manifest. Never FLUSHDB or touch other indexes.
                manifest[filename] = {'fingerprint': fingerprint, 'ids': identifiers}
                pipeline = self.client.pipeline(transaction=True)
                old_ids = set(previous['ids']) - set(identifiers) if previous else set()
                if old_ids:
                    pipeline.delete(*(self.prefix + ':' + i for i in old_ids))
                pipeline.set(self.meta_key, json.dumps(manifest, ensure_ascii=False))
                pipeline.execute()
                results.append({'filename': filename, 'status': 'updated', 'chunks': len(chunks)})
        return results

    def search(self, query):
        if not query.strip():
            raise ValueError('查询不能为空。')
        documents = self.vector_store.similarity_search(query, k=3)
        return [{**d.metadata, 'content': d.page_content} for d in documents]


def rebuild_project_index(client):
    """Explicit opt-in, project-only cleanup. No broad key deletion."""
    indexes = {v.decode() if isinstance(v, bytes) else v for v in client.execute_command('FT._LIST')}
    if INDEX in indexes:
        client.execute_command('FT.DROPINDEX', INDEX)  # no DD: only owned prefix below
    for key in client.scan_iter(match=INDEX + ':doc:*'):
        client.delete(key)
    client.delete(INDEX + ':config', INDEX + ':manifest')
