import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from charset_normalizer import from_bytes
from dotenv import load_dotenv


CHUNK_SIZE = 384
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\u3000\u3000", "\n", "。", " ", ""]
EMBEDDING_MODELS = (
    (
        "embed1.index",
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ),
    (
        "embed2.index",
        "richinfoai/ritrieve_zh_v1",
        "f8d5a707656c55705027678e311f9202c8ced12c",
    ),
)
PASSTHROUGH_EXTENSIONS = {".md", ".txt"}
FLASH_EXTENSIONS = {
    ".bmp",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".pdf",
    ".png",
    ".pptx",
    ".webp",
    ".xls",
    ".xlsx",
}
PRECISION_EXTENSIONS = FLASH_EXTENSIONS | {".doc", ".ppt"}
MINERU_EXTENSIONS = PRECISION_EXTENSIONS
CHUNK_FIELDS = ("chunk_id", "doc_id", "source", "section", "text")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_content_hash(path):
    path = Path(path)
    if path.suffix.lower() in PASSTHROUGH_EXTENSIONS:
        text = extract_text(path)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return sha256_file(path)


def stable_id(prefix, *parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def stable_document_id(document_hash):
    return stable_id("doc", document_hash)


def discover_documents(doc_dir):
    doc_dir = Path(doc_dir)
    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {doc_dir}")
    if not doc_dir.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {doc_dir}")

    supported_extensions = PASSTHROUGH_EXTENSIONS | MINERU_EXTENSIONS
    paths = sorted(
        (
            path
            for path in doc_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        ),
        key=lambda path: (
            path.relative_to(doc_dir).as_posix().casefold(),
            path.relative_to(doc_dir).as_posix(),
        ),
    )
    if not paths:
        raise ValueError(f"No supported documents found in: {doc_dir}")
    return paths


def split_sections(text):
    sections = []
    heading = None
    lines = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if any(part.strip() for part in lines):
                sections.append((heading, "\n".join(lines).strip()))
            heading = match.group(1).strip()
            lines = [line]
        else:
            lines.append(line)
    if any(part.strip() for part in lines):
        sections.append((heading, "\n".join(lines).strip()))
    return sections


class MineruClient:
    """Adapter around the official MinerU Open API SDK."""

    def __init__(
        self,
        token=None,
        client=None,
    ):
        self.token = token
        if client is None:
            from mineru import MinerU

            client = MinerU(token)
        self.client = client

    def extract(self, path):
        path = Path(path)
        if self.token:
            result = self.client.extract(str(path), model="vlm")
        else:
            result = self.client.flash_extract(str(path))
        if result.state != "done":
            raise RuntimeError(
                f"MinerU extraction failed ({result.state}): "
                f"{result.error or result.err_code or 'unknown error'}"
            )
        if not result.markdown:
            raise RuntimeError("MinerU returned empty Markdown")
        return result.markdown


def create_mineru_client():
    load_dotenv()
    return MineruClient(os.getenv("MINERU_API_TOKEN") or os.getenv("MINERU_TOKEN"))


def extract_text(path, mineru=None):
    try:
        suffix = path.suffix.lower()
        if suffix in PASSTHROUGH_EXTENSIONS:
            match = from_bytes(path.read_bytes()).best()
            if match is None:
                raise UnicodeError(f"Unable to detect text encoding for '{path}'")
            return str(match)
        if suffix not in MINERU_EXTENSIONS:
            raise RuntimeError(f"Unsupported document format for MinerU: '{suffix}'")
        if mineru is None:
            mineru = create_mineru_client()
        supported = (
            PRECISION_EXTENSIONS if getattr(mineru, "token", None) else FLASH_EXTENSIONS
        )
        if suffix not in supported:
            mode = "extract" if mineru.token else "flash_extract"
            raise RuntimeError(f"'{suffix}' is not supported by MinerU {mode}")
        return mineru.extract(path)
    except Exception as error:
        raise RuntimeError(f"Failed to parse document '{path}': {error}") from error


def normalize_text_files(doc_dir):
    converted = []
    for path in discover_documents(doc_dir):
        if path.suffix.lower() not in PASSTHROUGH_EXTENSIONS:
            continue
        raw = path.read_bytes()
        utf8 = extract_text(path).encode("utf-8")
        if raw == utf8:
            continue

        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".utf8.tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(utf8)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
            converted.append(path)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary).unlink(missing_ok=True)
            raise
    return converted


def create_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be at least zero and smaller than chunk_size"
        )

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    return splitter


def build_document_chunks(path, doc_dir, splitter, mineru=None, document_hash=None):
    path = Path(path)
    doc_dir = Path(doc_dir)
    source = path.relative_to(doc_dir).as_posix()
    raw_hash = sha256_file(path)
    document_hash = document_hash or document_content_hash(path)
    doc_id = stable_document_id(document_hash)
    chunks = []
    chunk_number = 0

    page_text = extract_text(path, mineru)
    for section, section_text in split_sections(page_text):
        for text in splitter.split_text(section_text):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": stable_id("chunk", doc_id, section, chunk_number, text),
                    "doc_id": doc_id,
                    "source": source,
                    "section": section,
                    "text": text,
                }
            )
            chunk_number += 1

    if chunk_number == 0:
        raise RuntimeError(f"No extractable text found in document '{source}'")
    if sha256_file(path) != raw_hash:
        raise RuntimeError(f"Document changed while indexing: '{source}'")
    return chunks, {
        "doc_id": doc_id,
        "source": source,
        "sha256": document_hash,
    }


def build_chunks(
    doc_dir, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, mineru=None
):
    doc_dir = Path(doc_dir)
    splitter = create_splitter(chunk_size, chunk_overlap)
    paths = discover_documents(doc_dir)
    if mineru is None and any(
        path.suffix.lower() in MINERU_EXTENSIONS for path in paths
    ):
        mineru = create_mineru_client()
    chunks = []
    documents = []
    seen_hashes = set()

    for path in paths:
        document_hash = document_content_hash(path)
        if document_hash in seen_hashes:
            raise ValueError(f"Duplicate document content is not supported: '{path}'")
        seen_hashes.add(document_hash)
        document_chunks, document = build_document_chunks(
            path,
            doc_dir,
            splitter,
            mineru=mineru,
            document_hash=document_hash,
        )
        chunks.extend(document_chunks)
        documents.append(document)

    return chunks, documents


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_save_index(model, texts, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        model.save_index(texts, temporary)
        with open(temporary, "rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_manifest(
    chunks,
    documents,
    chunk_size,
    chunk_overlap,
    artifacts,
):
    manifest = {
        "schema": {
            "name": "docbot-index",
            "version": 2,
            "chunk_fields": list(CHUNK_FIELDS),
        },
        "embedding_models": [
            {"index": artifact, "name": name, "revision": revision}
            for artifact, name, revision in EMBEDDING_MODELS
        ],
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": SEPARATORS,
        },
        "documents": documents,
        "counts": {"documents": len(documents), "chunks": len(chunks)},
        "artifacts": {
            name: {"path": Path(path).name, "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    return manifest


def _required_index_paths(index_dir):
    return tuple(
        Path(index_dir) / name
        for name in ("manifest.json", "chunks.json", "embed1.index", "embed2.index")
    )


def _load_existing_bundle(index_dir):
    if not all(path.exists() for path in _required_index_paths(index_dir)):
        return None
    try:
        from pipeline import load_index_bundle

        return load_index_bundle(index_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _index_configuration_matches(manifest, chunk_size, chunk_overlap):
    expected_models = [
        {"index": path, "name": name, "revision": revision}
        for path, name, revision in EMBEDDING_MODELS
    ]
    return (
        manifest.get("chunking")
        == {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": SEPARATORS,
        }
        and manifest.get("embedding_models") == expected_models
    )


def _current_document_snapshot(doc_dir):
    documents = []
    seen_hashes = set()
    for path in discover_documents(doc_dir):
        document_hash = document_content_hash(path)
        if document_hash in seen_hashes:
            raise ValueError(f"Duplicate document content is not supported: '{path}'")
        seen_hashes.add(document_hash)
        documents.append(
            {
                "path": path,
                "source": path.relative_to(doc_dir).as_posix(),
                "sha256": document_hash,
            }
        )
    return documents


def _incremental_plan(
    doc_dir,
    manifest,
    old_chunks,
    chunk_size,
    chunk_overlap,
):
    current_documents = _current_document_snapshot(doc_dir)
    old_documents_by_hash = {}
    for document in manifest["documents"]:
        document_hash = document["sha256"]
        if document_hash in old_documents_by_hash:
            raise ValueError(
                "Existing index contains duplicate document content; rebuild it"
            )
        old_documents_by_hash[document_hash] = document

    old_chunks_by_doc_id = {}
    for row, chunk in enumerate(old_chunks):
        old_chunks_by_doc_id.setdefault(chunk["doc_id"], []).append((row, chunk))

    changed = any(
        document["sha256"] not in old_documents_by_hash
        for document in current_documents
    )
    splitter = None
    mineru = None
    if changed:
        splitter = create_splitter(chunk_size, chunk_overlap)
        if any(
            document["path"].suffix.lower() in MINERU_EXTENSIONS
            for document in current_documents
            if document["sha256"] not in old_documents_by_hash
        ):
            mineru = create_mineru_client()

    chunks = []
    reuse_rows = []
    documents = []
    for current in current_documents:
        old_document = old_documents_by_hash.get(current["sha256"])
        if old_document is None:
            new_chunks, new_document = build_document_chunks(
                current["path"],
                doc_dir,
                splitter,
                mineru=mineru,
                document_hash=current["sha256"],
            )
            chunks.extend(new_chunks)
            reuse_rows.extend([None] * len(new_chunks))
            documents.append(new_document)
            continue

        old_rows = old_chunks_by_doc_id.get(old_document["doc_id"], [])
        if not old_rows:
            raise ValueError(
                f"Existing index has no chunks for document '{old_document['source']}'"
            )
        doc_id = stable_document_id(current["sha256"])
        for chunk_number, (row, old_chunk) in enumerate(old_rows):
            text = old_chunk["text"]
            chunks.append(
                {
                    "chunk_id": stable_id(
                        "chunk", doc_id, old_chunk["section"], chunk_number, text
                    ),
                    "doc_id": doc_id,
                    "source": current["source"],
                    "section": old_chunk["section"],
                    "text": text,
                }
            )
            reuse_rows.append(row)
        documents.append(
            {
                "doc_id": doc_id,
                "source": current["source"],
                "sha256": current["sha256"],
            }
        )

    metadata_changed = (
        chunks != old_chunks
        or documents != manifest["documents"]
        or "identity" in manifest
    )
    return chunks, documents, reuse_rows, metadata_changed


def _save_incremental_index(
    source_path,
    destination,
    chunks,
    reuse_rows,
    old_chunk_count,
    model_name,
    revision,
):
    if len(chunks) == old_chunk_count and reuse_rows == list(range(len(chunks))):
        shutil.copyfile(source_path, destination)
        return

    import faiss
    import numpy as np

    old_index = faiss.read_index(str(source_path))
    if old_index.ntotal != old_chunk_count:
        raise ValueError(
            f"Embedding index count does not match chunks.json: {source_path}"
        )
    old_vectors = np.asarray(
        old_index.reconstruct_n(0, old_index.ntotal), dtype="float32"
    )
    missing = [row for row, old_row in enumerate(reuse_rows) if old_row is None]
    if missing:
        from model import EmbeddingModel
        from tqdm import tqdm

        model = EmbeddingModel(model_name, revision=revision)
        batch_size = 32
        batches = range(0, len(missing), batch_size)
        new_vectors = np.concatenate(
            [
                model._embed_texts(
                    [chunks[row]["text"] for row in missing[start : start + batch_size]]
                )
                for start in tqdm(
                    batches,
                    desc=f"Embedding {model_name}",
                    unit="batch",
                )
            ],
            axis=0,
        )
    else:
        new_vectors = None

    vectors = np.empty((len(chunks), old_index.d), dtype="float32")
    missing_by_row = dict(zip(missing, new_vectors)) if new_vectors is not None else {}
    for row, old_row in enumerate(reuse_rows):
        vectors[row] = (
            old_vectors[old_row] if old_row is not None else missing_by_row[row]
        )
    new_index = faiss.IndexFlatL2(old_index.d)
    new_index.add(vectors)
    faiss.write_index(new_index, str(destination))


def create_index(
    doc_dir="doc",
    index_dir="index",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    force=False,
):
    doc_dir = Path(doc_dir)
    converted = normalize_text_files(doc_dir) if doc_dir.exists() else []
    for path in converted:
        print(f"Converted to UTF-8: {path}")

    existing = None if force else _load_existing_bundle(index_dir)
    index_dir = Path(index_dir)

    if existing is not None:
        old_manifest, old_chunks = existing
        if _index_configuration_matches(old_manifest, chunk_size, chunk_overlap):
            chunks, documents, reuse_rows, changed = _incremental_plan(
                doc_dir,
                old_manifest,
                old_chunks,
                chunk_size,
                chunk_overlap,
            )
            if not changed:
                return old_manifest

            with tempfile.TemporaryDirectory(
                dir=index_dir.parent, prefix=f".{index_dir.name}.incremental."
            ) as staging_directory:
                staging = Path(staging_directory)
                chunks_path = staging / "chunks.json"
                atomic_write_json(chunks_path, chunks)
                artifacts = {"chunks": chunks_path}
                for (
                    artifact,
                    model_name,
                    revision,
                ) in EMBEDDING_MODELS:
                    artifact_path = staging / artifact
                    _save_incremental_index(
                        index_dir / artifact,
                        artifact_path,
                        chunks,
                        reuse_rows,
                        len(old_chunks),
                        model_name,
                        revision,
                    )
                    artifacts[Path(artifact).stem] = artifact_path

                new_manifest = build_manifest(
                    chunks,
                    documents,
                    chunk_size,
                    chunk_overlap,
                    artifacts,
                )
                atomic_write_json(staging / "manifest.json", new_manifest)
                index_dir.mkdir(parents=True, exist_ok=True)
                for name in (
                    "chunks.json",
                    *(item[0] for item in EMBEDDING_MODELS),
                ):
                    os.replace(staging / name, index_dir / name)
                os.replace(staging / "manifest.json", index_dir / "manifest.json")
            return new_manifest

    chunks, documents = build_chunks(doc_dir, chunk_size, chunk_overlap)
    texts = [chunk["text"] for chunk in chunks]
    from model import EmbeddingModel

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=index_dir.parent, prefix=f".{index_dir.name}.build."
    ) as staging_directory:
        staging = Path(staging_directory)
        chunks_path = staging / "chunks.json"
        atomic_write_json(chunks_path, chunks)
        artifacts = {"chunks": chunks_path}
        for artifact, model_name, revision in EMBEDDING_MODELS:
            artifact_path = staging / artifact
            atomic_save_index(
                EmbeddingModel(model_name, revision=revision), texts, artifact_path
            )
            artifacts[Path(artifact).stem] = artifact_path

        manifest = build_manifest(
            chunks,
            documents,
            chunk_size,
            chunk_overlap,
            artifacts,
        )
        atomic_write_json(staging / "manifest.json", manifest)

        index_dir.mkdir(parents=True, exist_ok=True)
        for name in ("chunks.json", *(item[0] for item in EMBEDDING_MODELS)):
            os.replace(staging / name, index_dir / name)
        # Publish the new checksums only after every expensive build succeeds.
        os.replace(staging / "manifest.json", index_dir / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build Docbot document indexes")
    parser.add_argument("--doc-dir", default="doc")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild all index artifacts even when an existing index is current",
    )
    args = parser.parse_args()
    manifest = create_index(
        doc_dir=args.doc_dir,
        index_dir="index",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
    )
    print(
        f"Index ready: {manifest['counts']['chunks']} chunks from "
        f"{manifest['counts']['documents']} documents into index"
    )


if __name__ == "__main__":
    main()
