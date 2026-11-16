import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).with_name("dflash_own_head.py")


def load_script(reader):
    spec = importlib.util.spec_from_file_location("dflash_own_head_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"gguf": SimpleNamespace(GGUFReader=reader)}):
        spec.loader.exec_module(module)
    return module


class DflashOwnHeadTest(unittest.TestCase):
    def test_output_aliases_are_rejected_before_reading_sources(self):
        reader = mock.Mock(side_effect=AssertionError("source read before output alias rejection"))
        tool = load_script(reader)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            draft = root / "draft.gguf"
            targets = [root / "target-00001.gguf", root / "target-00002.gguf"]
            for path in [draft, *targets]:
                path.write_bytes(b"not a GGUF")
            aliases = [("draft", draft), ("target symlink", root / "target-link.gguf"),
                       ("target hardlink", root / "target-hardlink.gguf")]
            aliases[1][1].symlink_to(targets[0])
            os.link(targets[1], aliases[2][1])
            for name, output in aliases:
                with self.subTest(name=name), mock.patch.object(sys, "argv", [str(SCRIPT), "--draft", str(draft),
                                                                                 "--target", *(str(path) for path in targets),
                                                                                 "--out", str(output)]):
                    with self.assertRaisesRegex(SystemExit, "--out aliases"):
                        tool.main()
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
