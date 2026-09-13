import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from create_index import MineruClient, extract_text


class FakeSdkClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def extract(self, source, **options):
        self.calls.append((source, options))
        return self.result

    def flash_extract(self, source, **options):
        self.calls.append((source, options))
        return self.result


class MineruClientTest(unittest.TestCase):
    def test_sdk_extract_uses_vlm_and_returns_markdown(self):
        sdk = FakeSdkClient(
            SimpleNamespace(state="done", markdown="# 来自 MinerU", error=None, err_code="")
        )
        client = MineruClient("token", client=sdk)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.pdf"
            path.write_bytes(b"pdf")
            self.assertEqual(client.extract(path), "# 来自 MinerU")

        source, options = sdk.calls[0]
        self.assertEqual(source, str(path))
        self.assertEqual(options["model"], "vlm")

    def test_without_token_uses_flash_extract(self):
        sdk = FakeSdkClient(
            SimpleNamespace(state="done", markdown="# Flash", error=None, err_code="")
        )
        client = MineruClient(client=sdk)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.pdf"
            path.write_bytes(b"pdf")
            self.assertEqual(client.extract(path), "# Flash")

        source, options = sdk.calls[0]
        self.assertEqual(source, str(path))
        self.assertEqual(options, {})

    def test_sdk_failed_result_is_rejected(self):
        sdk = FakeSdkClient(
            SimpleNamespace(state="failed", markdown=None, error="bad file", err_code="E1")
        )
        client = MineruClient("token", client=sdk)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.pdf"
            path.write_bytes(b"pdf")
            with self.assertRaisesRegex(RuntimeError, "bad file"):
                client.extract(path)

    def test_flash_rejects_formats_only_supported_by_precision_api(self):
        sdk = FakeSdkClient(
            SimpleNamespace(state="done", markdown="# Flash", error=None, err_code="")
        )
        client = MineruClient(client=sdk)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.doc"
            path.write_bytes(b"doc")
            with self.assertRaisesRegex(RuntimeError, "not supported by MinerU flash_extract"):
                extract_text(path, client)


if __name__ == "__main__":
    unittest.main()
