import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import create_index


class FakeMineru:
    def extract(self, path):
        return Path(path).read_text(encoding="utf-8")


class IndexMetadataTest(unittest.TestCase):
    def test_document_order_hashes_and_chunk_ids_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.md").write_text("# B\nbeta", encoding="utf-8")
            (root / "a.md").write_text("# A\nalpha", encoding="utf-8")

            paths = create_index.discover_documents(root)
            self.assertEqual([path.name for path in paths], ["a.md", "b.md"])
            digest = create_index.sha256_file(paths[0])
            self.assertEqual(
                digest,
                "5d1b246525f1b02d8f6a8dc450a56aecced46989009ca9070ced2238562b70e1",
            )
            self.assertEqual(
                create_index.stable_document_id(digest),
                create_index.stable_document_id(digest),
            )
            self.assertEqual(
                create_index.stable_document_id(digest),
                create_index.stable_document_id(digest),
            )

    def test_chunk_ids_ignore_filename_but_source_keeps_current_name(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            (first_root / "original.md").write_text("# Guide\ncontent", encoding="utf-8")
            (second_root / "renamed.md").write_text("# Guide\ncontent", encoding="utf-8")

            first_chunks, first_documents = create_index.build_chunks(first_root)
            second_chunks, second_documents = create_index.build_chunks(second_root)

            self.assertEqual(first_documents[0]["doc_id"], second_documents[0]["doc_id"])
            self.assertEqual(
                [chunk["chunk_id"] for chunk in first_chunks],
                [chunk["chunk_id"] for chunk in second_chunks],
            )
            self.assertEqual(first_chunks[0]["source"], "original.md")
            self.assertEqual(second_chunks[0]["source"], "renamed.md")

    def test_manifest_contains_reproducible_metadata_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                {
                    "chunk_id": "chunk_1",
                    "doc_id": "doc_1",
                    "source": "sample.md",
                    "section": "Sample",
                    "text": "evidence",
                }
            ]
            documents = [
                {"doc_id": "doc_1", "source": "sample.md", "sha256": "abc"}
            ]
            chunks_path = root / "chunks.json"
            create_index.atomic_write_json(chunks_path, chunks)
            manifest = create_index.build_manifest(
                chunks, documents, 512, 200, {"chunks": chunks_path}
            )

            self.assertEqual(
                manifest["schema"]["chunk_fields"], list(create_index.CHUNK_FIELDS)
            )
            self.assertEqual(manifest["chunking"]["chunk_overlap"], 200)
            self.assertEqual(manifest["counts"], {"documents": 1, "chunks": 1})
            self.assertEqual(manifest["documents"], documents)
            self.assertEqual(
                manifest["embedding_models"][0],
                {
                    "index": "embed1.index",
                    "name": "Qwen/Qwen3-Embedding-0.6B",
                    "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                },
            )
            self.assertEqual(
                manifest["artifacts"]["chunks"]["sha256"],
                create_index.sha256_file(chunks_path),
            )
            self.assertEqual(manifest["artifacts"]["chunks"]["path"], "chunks.json")
            self.assertEqual(
                json.loads(chunks_path.read_text(encoding="utf-8")), chunks
            )

    def test_empty_and_unsupported_inputs_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No supported documents found"):
                create_index.discover_documents(root)
            (root / "binary.exe").write_bytes(b"not a document")
            with self.assertRaisesRegex(ValueError, "No supported documents found"):
                create_index.discover_documents(root)

            (root / "supported.txt").write_text("可识别", encoding="utf-8")
            self.assertEqual([path.name for path in create_index.discover_documents(root)], ["supported.txt"])

    def test_text_encoding_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.txt"
            path.write_bytes("中文文本：自动识别 GB18030。".encode("gb18030"))
            self.assertEqual(create_index.extract_text(path), "中文文本：自动识别 GB18030。")

    def test_normalize_text_files_overwrites_legacy_text_as_utf8(self):
        text = "中文文本：转换为 UTF-8。"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.txt"
            path.write_bytes(text.encode("gb18030"))

            converted = create_index.normalize_text_files(root)

            self.assertEqual(converted, [path])
            self.assertEqual(path.read_bytes(), text.encode("utf-8"))

    def test_text_encoding_does_not_change_document_identity(self):
        text = "# 标题\n中文文本：UTF-8 和 GB18030 应生成相同 ID。"
        with tempfile.TemporaryDirectory() as utf8_dir, tempfile.TemporaryDirectory() as legacy_dir:
            utf8_root = Path(utf8_dir)
            legacy_root = Path(legacy_dir)
            (utf8_root / "book.txt").write_bytes(text.encode("utf-8"))
            (legacy_root / "book.txt").write_bytes(text.encode("gb18030"))

            utf8_chunks, utf8_documents = create_index.build_chunks(utf8_root)
            legacy_chunks, legacy_documents = create_index.build_chunks(legacy_root)

            self.assertEqual(
                utf8_documents[0]["doc_id"], legacy_documents[0]["doc_id"]
            )
            self.assertEqual(
                utf8_documents[0]["sha256"], legacy_documents[0]["sha256"]
            )
            self.assertEqual(
                [chunk["chunk_id"] for chunk in utf8_chunks],
                [chunk["chunk_id"] for chunk in legacy_chunks],
            )

    def test_parse_failures_name_the_document(self):
        class BrokenMineru:
            def extract(self, path):
                raise ValueError("broken input")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pdf"
            path.write_text("content", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, r"Failed to parse document '.*broken\.pdf': broken input"
            ):
                create_index.extract_text(path, BrokenMineru())

    def test_long_unbroken_text_still_respects_chunk_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "long.txt").write_text("字" * 1200, encoding="utf-8")

            chunks, _ = create_index.build_chunks(
                root, chunk_size=512, chunk_overlap=100, mineru=FakeMineru()
            )

            self.assertGreater(len(chunks), 1)
            self.assertLessEqual(max(len(chunk["text"]) for chunk in chunks), 512)

    def test_document_change_during_indexing_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "changing.txt").write_text("content", encoding="utf-8")

            with patch(
                "create_index.sha256_file", side_effect=("before", "after")
            ), self.assertRaisesRegex(RuntimeError, "changed while indexing"):
                create_index.build_chunks(root, mineru=FakeMineru())

    def test_failed_rebuild_preserves_previous_bundle(self):
        class FirstEmbedding:
            def save_index(self, texts, path):
                Path(path).write_bytes(b"new index")

        chunk_id = "chunk_" + "a" * 64
        doc_id = "doc_" + "b" * 64
        chunks = [
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source": "sample.md",
                "section": None,
                "text": "new text",
            }
        ]
        documents = [
            {"doc_id": doc_id, "source": "sample.md", "sha256": "c" * 64}
        ]
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory) / "index"
            index_dir.mkdir()
            previous = {
                "chunks.json": b"old chunks",
                "embed1.index": b"old one",
                "embed2.index": b"old two",
                "manifest.json": b"old manifest",
            }
            for name, content in previous.items():
                (index_dir / name).write_bytes(content)

            with patch(
                "create_index.build_chunks", return_value=(chunks, documents)
            ), patch(
                "model.EmbeddingModel",
                side_effect=(FirstEmbedding(), RuntimeError("second model failed")),
            ), self.assertRaisesRegex(RuntimeError, "second model failed"):
                create_index.create_index("unused", index_dir)

            self.assertEqual(
                {name: (index_dir / name).read_bytes() for name in previous},
                previous,
            )

    def test_incremental_index_skips_unchanged_documents(self):
        class FakeEmbedding:
            def __init__(self, name, revision=None):
                self.name = name
                self.revision = revision

            def save_index(self, texts, path):
                import faiss
                import numpy as np

                index = faiss.IndexFlatL2(2)
                vectors = np.asarray(
                    [[len(text), index_number] for index_number, text in enumerate(texts)],
                    dtype="float32",
                )
                index.add(vectors)
                faiss.write_index(index, str(path))

            def _embed_texts(self, texts):
                import numpy as np

                return np.asarray(
                    [[len(text), index_number] for index_number, text in enumerate(texts)],
                    dtype="float32",
                )

        with tempfile.TemporaryDirectory() as directory, patch(
            "model.EmbeddingModel", FakeEmbedding
        ):
            root = Path(directory)
            doc_dir = root / "docs"
            index_dir = root / "index"
            doc_dir.mkdir()
            (doc_dir / "guide.md").write_text("# Guide\ncontent", encoding="utf-8")
            create_index.create_index(doc_dir, index_dir)

            with patch(
                "model.EmbeddingModel",
                side_effect=AssertionError("unchanged index loaded a model"),
            ):
                create_index.create_index(doc_dir, index_dir)

    def test_index_configuration_does_not_require_identity_strategy(self):
        manifest = {
            "chunking": {
                "chunk_size": create_index.CHUNK_SIZE,
                "chunk_overlap": create_index.CHUNK_OVERLAP,
                "separators": create_index.SEPARATORS,
            },
            "embedding_models": [
                {
                    "index": path,
                    "name": name,
                    "revision": revision,
                }
                for path, name, revision in create_index.EMBEDDING_MODELS
            ],
        }
        self.assertTrue(
            create_index._index_configuration_matches(
                manifest,
                create_index.CHUNK_SIZE,
                create_index.CHUNK_OVERLAP,
            )
        )

    def test_incremental_index_updates_source_on_rename_without_embedding(self):
        class FakeEmbedding:
            def __init__(self, name, revision=None):
                pass

            def save_index(self, texts, path):
                import faiss
                import numpy as np

                index = faiss.IndexFlatL2(2)
                index.add(np.ones((len(texts), 2), dtype="float32"))
                faiss.write_index(index, str(path))

            def _embed_texts(self, texts):
                raise AssertionError("rename should reuse existing vectors")

        with tempfile.TemporaryDirectory() as directory, patch(
            "model.EmbeddingModel", FakeEmbedding
        ):
            root = Path(directory)
            doc_dir = root / "docs"
            index_dir = root / "index"
            doc_dir.mkdir()
            original = doc_dir / "original.md"
            original.write_text("# Guide\ncontent", encoding="utf-8")
            create_index.create_index(doc_dir, index_dir)
            before = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

            original.rename(doc_dir / "renamed.md")
            create_index.create_index(doc_dir, index_dir)
            after = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

            self.assertEqual(
                [chunk["chunk_id"] for chunk in before],
                [chunk["chunk_id"] for chunk in after],
            )
            self.assertEqual({chunk["source"] for chunk in after}, {"renamed.md"})

    def test_incremental_index_embeds_only_new_document(self):
        class FakeEmbedding:
            calls = []

            def __init__(self, name, revision=None):
                self.name = name

            @staticmethod
            def vectors(texts):
                import numpy as np

                return np.asarray(
                    [[len(text), 0] for text in texts], dtype="float32"
                )

            def save_index(self, texts, path):
                import faiss

                index = faiss.IndexFlatL2(2)
                index.add(self.vectors(texts))
                faiss.write_index(index, str(path))

            def _embed_texts(self, texts):
                type(self).calls.append((self.name, list(texts)))
                return self.vectors(texts)

        with tempfile.TemporaryDirectory() as directory, patch(
            "model.EmbeddingModel", FakeEmbedding
        ):
            root = Path(directory)
            doc_dir = root / "docs"
            index_dir = root / "index"
            doc_dir.mkdir()
            (doc_dir / "first.md").write_text("# First\none", encoding="utf-8")
            create_index.create_index(doc_dir, index_dir)
            FakeEmbedding.calls.clear()

            (doc_dir / "second.md").write_text("# Second\ntwo", encoding="utf-8")
            create_index.create_index(doc_dir, index_dir)

            self.assertEqual(
                FakeEmbedding.calls,
                [("Qwen/Qwen3-Embedding-0.6B", ["# Second\ntwo"]),
                 ("richinfoai/ritrieve_zh_v1", ["# Second\ntwo"])],
            )


if __name__ == "__main__":
    unittest.main()
