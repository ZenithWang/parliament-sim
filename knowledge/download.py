"""Explicit download only. Original PDFs remain outside version control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = Path(__file__).with_name('sources.json')


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'sources.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for source in json.loads(SOURCES.read_text()):
        path = destination / source['filename']
        if not path.exists():
            request = urllib.request.Request(source['url'], headers={'User-Agent': 'ParliamentSim/2.0 (educational document retrieval)'})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read(30 * 1024 * 1024 + 1)
            if len(body) > 30 * 1024 * 1024 or not body.startswith(b'%PDF'):
                raise ValueError('下载不是有效 PDF 或超过 30MB。')
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(body)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        old = next((m for m in manifest if m['filename'] == source['filename']), None)
        entry = {**source, 'sha256': digest, 'downloaded_at': old['downloaded_at'] if old and old.get('sha256') == digest else datetime.now(timezone.utc).isoformat()}
        manifest = [m for m in manifest if m['filename'] != source['filename']] + [entry]
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(manifest_path)
        print(f"已准备：{path}")


def main():
    parser = argparse.ArgumentParser(description='显式下载两份公开报告，保存来源清单。')
    parser.add_argument('--output', default='data/documents')
    args = parser.parse_args()
    download(args.output)


if __name__ == '__main__':
    main()
