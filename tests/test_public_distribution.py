from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {
    ".cmd",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".txt",
}
PUBLIC_TEXT_NAMES = {".gitattributes", ".gitignore", "LICENSE"}


def distributable_files() -> list[Path]:
    """Return tracked and not-ignored files without descending into local assets."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    relative_paths = result.stdout.decode("utf-8").split("\0")
    return [ROOT / relative for relative in relative_paths if relative]


class PublicDistributionTests(unittest.TestCase):
    def test_distributable_text_has_no_developer_specific_absolute_paths(self) -> None:
        # Keep the forbidden values assembled so this test does not flag itself.
        forbidden = (
            "c:" + "\\users\\" + "niku_",
            "c:" + "/users/" + "niku_",
            "c:" + "\\project_hub",
            "c:" + "/project_hub",
        )
        violations: list[str] = []

        for path in distributable_files():
            if path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES and path.name not in PUBLIC_TEXT_NAMES:
                continue
            text = path.read_text(encoding="utf-8").casefold()
            if any(value in text for value in forbidden):
                violations.append(path.relative_to(ROOT).as_posix())

        self.assertEqual([], violations, f"developer-specific paths found in: {violations}")

    def test_setup_scripts_do_not_default_to_a_specific_python_executable(self) -> None:
        declaration = re.compile(
            r"\[\s*string\s*\]\s*\$PythonExe(?:\s*=\s*(?P<default>[^,\r\n\)]*))?",
            re.IGNORECASE,
        )
        for relative in ("scripts/setup.ps1", "scripts/setup_comfy.ps1"):
            with self.subTest(script=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                match = declaration.search(text)
                self.assertIsNotNone(match, f"{relative} must expose a PythonExe parameter")
                default = (match.group("default") or "").strip().casefold()
                self.assertIn(default, {"", '""', "''", "$null"})

    def test_windows_launchers_are_location_independent(self) -> None:
        for relative in ("Setup-H3-Studio.cmd", "Start-H3-WebUI.cmd"):
            with self.subTest(launcher=relative):
                launcher = ROOT / relative
                self.assertTrue(launcher.is_file(), f"missing launcher: {relative}")
                self.assertIn("%~dp0", launcher.read_text(encoding="utf-8").casefold())

    def test_gitignore_excludes_local_runtime_and_large_artifacts(self) -> None:
        patterns = {
            line.strip().replace("\\", "/").casefold()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        required = {
            ".venv/",
            ".comfy-venv/",
            "models/",
            "outputs/",
            "webui_data/",
            "*.safetensors",
            "*.ckpt",
            "*.pt",
            "*.pth",
        }
        self.assertFalse(required - patterns, f"missing ignore rules: {sorted(required - patterns)}")

    def test_comfy_model_lock_is_complete_and_uses_an_immutable_license_url(self) -> None:
        lock = json.loads((ROOT / "comfy_models.lock.json").read_text(encoding="utf-8"))
        files = lock["files"]

        self.assertEqual(5, len(files))
        self.assertEqual(63_440_965_087, lock["verification"]["total_bytes"])
        self.assertEqual(lock["verification"]["total_bytes"], sum(item["size"] for item in files))
        self.assertEqual(5, len({item["path"] for item in files}))
        for item in files:
            with self.subTest(model=item["path"]):
                self.assertTrue(item["path"].startswith("models/comfy/"))
                self.assertTrue(item["path"].endswith(".safetensors"))
                self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")

        source = lock["source"]
        self.assertEqual("minimax-h3-community-license-agreement", source["license"])
        self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
        self.assertRegex(
            source["license_url"],
            r"^https://huggingface\.co/MiniMaxAI/MiniMax-H3/blob/[0-9a-f]{40}/LICENSE$",
        )

    def test_repository_line_ending_policy_is_explicit(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertRegex(attributes, r"(?m)^\* text=auto eol=lf$")
        self.assertRegex(attributes, r"(?m)^\*\.cmd text eol=crlf$")


if __name__ == "__main__":
    unittest.main()
