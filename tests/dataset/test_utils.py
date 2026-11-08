import gzip
import io
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from pactus import config
from pactus.dataset import _utils

URL = "https://example.test/data/sample.zip"


class FakeResponse:
    def __init__(self, content: bytes, status: int = 200, headers: dict | None = None) -> None:
        self.content, self.status_code = content, status
        self.headers = {"content-length": str(len(content))} if headers is None else headers

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


def _serve(monkeypatch: pytest.MonkeyPatch, response: FakeResponse) -> None:
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: response)


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as file:
        file.writestr("inner.txt", "content")
    return buffer.getvalue()


def _folder() -> Path:
    return _utils._get_path(config.DS_DIR, "demo")


@pytest.mark.parametrize(
    ("part", "total", "expected"),
    [(1048576, -1, "Downloaded: 1.00 MB"), (5, 10, "50.00%"), (10, 10, "100.00%\n")],
)
def test_progress_text(part: int, total: int, expected: str) -> None:
    assert expected in _utils._get_progress_log(part, total)


def test_only_compressed_files_are_accepted() -> None:
    with pytest.raises(ValueError, match=r"zip or \.gz"):
        _utils.download_dataset("https://example.test/data.txt", "demo")


def test_a_failed_request_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(b"", status=404))
    with pytest.raises(RuntimeError, match="demo"):
        _utils.download_dataset(URL, "demo")


def test_zip_files_are_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(_zip_bytes()))
    _utils.download_dataset(URL, "demo")
    assert (_folder() / "inner.txt").read_text() == "content"


def test_gz_files_are_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(gzip.compress(b"content")))
    _utils.download_dataset("https://example.test/sample.txt.gz", "demo")
    assert (_folder() / "sample.txt").read_text() == "content"


def test_tar_files_are_extracted(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as file:
        info = tarfile.TarInfo("inner.txt")
        info.size = 7
        file.addfile(info, io.BytesIO(b"content"))
    _serve(monkeypatch, FakeResponse(buffer.getvalue()))
    _utils.download_dataset("https://example.test/sample.tar.gz", "demo")
    assert (_folder() / "inner.txt").read_text() == "content"


def test_a_complete_file_is_not_downloaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(monkeypatch, FakeResponse(_zip_bytes()))
    _utils.download_dataset(URL, "demo")
    written = []
    monkeypatch.setattr(_utils, "_download_until_finish", lambda *a: written.append(a))
    _utils.download_dataset(URL, "demo")
    assert written == []


@pytest.mark.parametrize(("answer", "replaced"), [("y", True), ("n", False)])
@pytest.mark.parametrize("headers", [None, {}], ids=["known size", "unknown size"])
def test_a_different_file_is_replaced_only_when_confirmed(
    monkeypatch: pytest.MonkeyPatch, answer: str, replaced: bool, headers: dict | None
) -> None:
    content = _zip_bytes()
    folder = _folder()
    folder.mkdir(parents=True)
    (folder / "sample.zip").write_bytes(b"old")
    _serve(monkeypatch, FakeResponse(content, headers=headers))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    monkeypatch.setattr(_utils, "_uncompress", lambda *a: None)
    _utils.download_dataset(URL, "demo")
    assert ((folder / "sample.zip").read_bytes() == content) is replaced
