import argparse
import json
import sys

from knowledge.documents import collect_documents
from knowledge.store import KnowledgeStore, create_client, create_embeddings, rebuild_project_index
from security import safe_error


def main():
    parser = argparse.ArgumentParser(description='导入 PDF/Markdown 到本项目 Redis 向量索引')
    parser.add_argument('--input', default='data/documents')
    parser.add_argument('--rebuild', action='store_true', help='仅重建 parliament_sim_v2 索引和所属文档')
    args = parser.parse_args()
    try:
        files = collect_documents(args.input)
        client, embeddings = create_client(), create_embeddings()
        if args.rebuild:
            rebuild_project_index(client)
        store = KnowledgeStore(embeddings, client)
        print(json.dumps(store.ingest(files), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
