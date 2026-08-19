"""
watch_folder.py - DocuLens AI auto-ingest folder watcher

Watches a folder for new PNG/JPG/JPEG/PDF files and automatically uploads
each one to a running DocuLens AI server. Useful for a "set it and forget
it" demo: drop screenshots into the watched folder and they get processed
without touching the dashboard.

Requires the DocuLens server to already be running (see README).

Usage:
    python watch_folder.py
    python watch_folder.py --folder ~/Desktop/DoculensInbox --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self._recently_seen = set()

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(Path(event.src_path))

    def on_moved(self, event):
        if event.is_directory:
            return
        self._handle(Path(event.dest_path))

    def _handle(self, path: Path):
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return
        if path in self._recently_seen:
            return
        self._recently_seen.add(path)

        # Give the OS a moment to finish writing the file (e.g. screenshots)
        time.sleep(0.5)
        if not path.exists():
            return

        print(f"[watch_folder] New file detected: {path.name} - uploading...")
        try:
            with open(path, "rb") as f:
                response = requests.post(
                    f"{self.server_url}/upload",
                    files={"file": (path.name, f)},
                    timeout=120,
                )
            if response.ok:
                data = response.json()
                print(
                    f"[watch_folder] ✓ Processed '{path.name}' as "
                    f"{data.get('document_type')} (confidence "
                    f"{data.get('confidence')}, risk {data.get('risk', {}).get('risk_level')})"
                )
            else:
                print(f"[watch_folder] ✗ Server returned {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print(
                f"[watch_folder] ✗ Could not reach {self.server_url}. "
                "Is the DocuLens server running?"
            )
        except Exception as e:
            print(f"[watch_folder] ✗ Failed to upload {path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto-ingest new documents into DocuLens AI.")
    parser.add_argument(
        "--folder",
        default=str(Path(__file__).resolve().parent / "data" / "watch_inbox"),
        help="Folder to watch for new documents (default: data/watch_inbox)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running DocuLens server (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    watch_path = Path(args.folder).expanduser()
    watch_path.mkdir(parents=True, exist_ok=True)

    print(f"[watch_folder] Watching: {watch_path}")
    print(f"[watch_folder] Uploading to: {args.url}")
    print("[watch_folder] Drop a PNG, JPG, JPEG or PDF into this folder to auto-process it.")
    print("[watch_folder] Press Ctrl+C to stop.\n")

    handler = NewFileHandler(args.url)
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[watch_folder] Stopped.")
    observer.join()


if __name__ == "__main__":
    main()
