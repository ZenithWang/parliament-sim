import argparse
import json
import sys
from knowledge.store import KnowledgeStore
from security import safe_error


def main():
    parser = argparse.ArgumentParser(description='独立验证本地知识库检索')
    parser.add_argument('--query', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(KnowledgeStore.open_for_search().search(args.query), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(safe_error(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
