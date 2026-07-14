from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CANONICAL_FILES = {
    "scheduler/strategy.py",
    "scheduler/graph.py",
    "scheduler/types.py",
    "scheduler/onnx_importer.py",
    "scheduler/hardware.py",
    "scheduler/autotune.py",
    "scheduler/benchmark.py",
    "scheduler/tuning_cache.py",
    "scheduler/graph_passes/__init__.py",
    "scheduler/graph_passes/fusion.py",
    "scheduler/graph_passes/pipeline.py",
    "scheduler/__init__.py",
}

TEMP_NAME_PATTERNS = (
    re.compile(r".*strategy[_-](fixed|compatible|old|backup|bak|copy).*\.py$", re.I),
    re.compile(r".*passes[_-](old|backup|bak|copy).*\.py$", re.I),
    re.compile(r".*\.(orig|rej|bak|tmp)$", re.I),
    re.compile(r".*~$", re.I),
    re.compile(r".*patch.*\.(zip|tar|gz|tgz)$", re.I),
)

ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
)

TEST_LEAK_PATTERNS = (
    re.compile(r"\bpytest\b"),
    re.compile(r"\bunittest\b"),
    re.compile(r"\btests?[\\/]", re.I),
    re.compile(r"release_to_competitors"),
    re.compile(r"testcases"),
)

EXPECTED_PUBLIC_IMPORTS = {
    "scheduler": (
        "Graph",
        "GraphNode",
        "HardwareSpec",
        "KernelSpecRef",
        "KernelTuningParams",
        "PrecisionProfile",
        "ProblemSize",
        "SchedulingStrategy",
        "import_onnx_graph",
    ),
    "scheduler.graph_passes": (
        "FusionPass",
        "GraphPassPipeline",
    ),
}


@dataclass
class Finding:
    severity: str
    category: str
    path: str
    message: str


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def find_cleanup_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = rel(path, root)

        if relative in CANONICAL_FILES:
            continue

        name = path.name
        if any(pattern.match(name) for pattern in TEMP_NAME_PATTERNS):
            candidates.append(path)
            continue

        # Duplicate strategy/pass files outside the canonical location.
        if name == "strategy.py" and relative != "scheduler/strategy.py":
            candidates.append(path)
        elif name == "passes.py":
            # scheduler/passes.py is obsolete only when graph_passes is present.
            if relative == "scheduler/passes.py" and (
                root / "scheduler/graph_passes"
            ).is_dir():
                candidates.append(path)
            elif relative != "scheduler/passes.py":
                candidates.append(path)

    return sorted(set(candidates))


def clean_candidates(
    root: Path,
    candidates: list[Path],
    apply: bool,
) -> list[str]:
    actions: list[str] = []
    for path in candidates:
        relative = rel(path, root)
        if apply:
            path.unlink()
            actions.append(f"deleted: {relative}")
        else:
            actions.append(f"would delete: {relative}")

    # Remove Python caches only in apply mode.
    for cache in root.rglob("__pycache__"):
        if cache.is_dir():
            if apply:
                shutil.rmtree(cache)
                actions.append(f"deleted cache: {rel(cache, root)}")
            else:
                actions.append(f"would delete cache: {rel(cache, root)}")

    for pyc in root.rglob("*.pyc"):
        if pyc.is_file():
            if apply:
                pyc.unlink()
                actions.append(f"deleted bytecode: {rel(pyc, root)}")
            else:
                actions.append(f"would delete bytecode: {rel(pyc, root)}")

    return actions


def syntax_check(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted((root / "scheduler").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            findings.append(
                Finding(
                    "error",
                    "syntax",
                    rel(path, root),
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return findings


def import_path_check(root: Path) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    details: dict[str, Any] = {}

    old_cwd = Path.cwd()
    old_path = list(sys.path)

    try:
        os.chdir(root)
        sys.path.insert(0, str(root))

        for module_name, expected_names in EXPECTED_PUBLIC_IMPORTS.items():
            try:
                module = importlib.import_module(module_name)
                details[module_name] = {
                    "file": getattr(module, "__file__", None),
                    "exports": {},
                }
            except Exception as exc:
                findings.append(
                    Finding(
                        "error",
                        "import",
                        module_name,
                        f"Cannot import module: {type(exc).__name__}: {exc}",
                    )
                )
                continue

            module_file = Path(module.__file__).resolve()
            try:
                module_file.relative_to(root.resolve())
            except ValueError:
                findings.append(
                    Finding(
                        "error",
                        "import",
                        module_name,
                        f"Imported from outside submission: {module_file}",
                    )
                )

            for name in expected_names:
                exists = hasattr(module, name)
                details[module_name]["exports"][name] = exists
                if not exists:
                    findings.append(
                        Finding(
                            "error",
                            "public-api",
                            module_name,
                            f"Missing public object: {name}",
                        )
                    )

        try:
            from scheduler.strategy import SchedulingStrategy
            signature = inspect.signature(SchedulingStrategy.__init__)
            details["SchedulingStrategy.__init__"] = str(signature)
            required = {"hardware", "full_fp32", "autotune_mode", "tuning_cache_path"}
            missing = required - set(signature.parameters)
            if missing:
                findings.append(
                    Finding(
                        "error",
                        "public-api",
                        "scheduler/strategy.py",
                        f"SchedulingStrategy.__init__ missing: {sorted(missing)}",
                    )
                )
        except Exception as exc:
            findings.append(
                Finding(
                    "error",
                    "public-api",
                    "scheduler/strategy.py",
                    f"Cannot inspect SchedulingStrategy: {type(exc).__name__}: {exc}",
                )
            )

        try:
            from scheduler.graph_passes import GraphPassPipeline
            signature = inspect.signature(GraphPassPipeline.__init__)
            details["GraphPassPipeline.__init__"] = str(signature)
            if "enable_fusion" not in signature.parameters:
                findings.append(
                    Finding(
                        "error",
                        "public-api",
                        "scheduler/graph_passes/pipeline.py",
                        "GraphPassPipeline.__init__ lacks enable_fusion",
                    )
                )
        except Exception as exc:
            findings.append(
                Finding(
                    "error",
                    "public-api",
                    "scheduler/graph_passes",
                    f"Cannot inspect GraphPassPipeline: {type(exc).__name__}: {exc}",
                )
            )

    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for module_name in list(sys.modules):
            if module_name == "scheduler" or module_name.startswith("scheduler."):
                sys.modules.pop(module_name, None)

    return findings, details


def inspect_scheduler_init(root: Path) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    init_path = root / "scheduler/__init__.py"
    details: dict[str, Any] = {}

    if not init_path.is_file():
        return [
            Finding(
                "error",
                "public-api",
                "scheduler/__init__.py",
                "Missing scheduler/__init__.py",
            )
        ], details

    content = init_path.read_text(encoding="utf-8")
    details["content"] = content

    try:
        tree = ast.parse(content)
    except Exception as exc:
        return [
            Finding(
                "error",
                "syntax",
                "scheduler/__init__.py",
                f"{type(exc).__name__}: {exc}",
            )
        ], details

    imported_names: set[str] = set()
    declared_all: list[str] | None = None

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ):
            try:
                value = ast.literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    declared_all = [str(item) for item in value]
            except Exception:
                pass

    details["imported_names"] = sorted(imported_names)
    details["__all__"] = declared_all

    expected = set(EXPECTED_PUBLIC_IMPORTS["scheduler"])
    missing_imports = expected - imported_names
    if missing_imports:
        findings.append(
            Finding(
                "error",
                "public-api",
                "scheduler/__init__.py",
                f"Missing imports: {sorted(missing_imports)}",
            )
        )

    if declared_all is None:
        findings.append(
            Finding(
                "warning",
                "public-api",
                "scheduler/__init__.py",
                "__all__ is not declared",
            )
        )
    else:
        missing_all = expected - set(declared_all)
        if missing_all:
            findings.append(
                Finding(
                    "error",
                    "public-api",
                    "scheduler/__init__.py",
                    f"__all__ missing: {sorted(missing_all)}",
                )
            )

    return findings, details


def source_leak_check(root: Path) -> list[Finding]:
    findings: list[Finding] = []

    for path in sorted((root / "scheduler").rglob("*.py")):
        content = path.read_text(encoding="utf-8", errors="replace")
        relative = rel(path, root)

        for pattern in ABSOLUTE_PATH_PATTERNS:
            match = pattern.search(content)
            if match:
                findings.append(
                    Finding(
                        "error",
                        "absolute-path",
                        relative,
                        f"Absolute path fragment: {match.group(0)!r}",
                    )
                )

        for pattern in TEST_LEAK_PATTERNS:
            match = pattern.search(content)
            if match:
                findings.append(
                    Finding(
                        "warning",
                        "test-leak",
                        relative,
                        f"Test-only reference in production code: {match.group(0)!r}",
                    )
                )

        if "if __name__ == \"__main__\"" in content or "if __name__ == '__main__'" in content:
            findings.append(
                Finding(
                    "warning",
                    "test-leak",
                    relative,
                    "Production module contains a __main__ execution block",
                )
            )

    return findings


def run_test_suite(root: Path) -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    ]

    results: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
        )
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )

    return {
        "passed": all(item["passed"] for item in results),
        "results": results,
    }


def generate_manifest(root: Path) -> list[dict[str, Any]]:
    excluded_dirs = {".git", ".pytest_cache", "__pycache__"}
    excluded_suffixes = {".pyc", ".pyo"}

    manifest: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.suffix in excluded_suffixes:
            continue

        relative = rel(path, root)
        stat = path.stat()
        manifest.append(
            {
                "path": relative,
                "size_bytes": stat.st_size,
                "category": (
                    "production"
                    if relative.startswith("scheduler/")
                    else "test"
                    if relative.startswith("tests/")
                    else "report-or-tool"
                ),
            }
        )
    return manifest


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# C3 final submission audit",
        "",
        f"- Overall passed: **{report['summary']['passed']}**",
        f"- Errors: **{report['summary']['errors']}**",
        f"- Warnings: **{report['summary']['warnings']}**",
        f"- Test suite passed: **{report['test_suite']['passed']}**",
        "",
        "## Cleanup",
        "",
    ]

    actions = report["cleanup"]["actions"]
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- No duplicate or temporary files found.")

    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(
                f"- **{finding['severity'].upper()}** "
                f"`{finding['category']}` `{finding['path']}`: "
                f"{finding['message']}"
            )
    else:
        lines.append("- No findings.")

    lines.extend(["", "## Public imports", ""])
    for module, details in report["imports"].items():
        if isinstance(details, dict) and "exports" in details:
            lines.append(f"### `{module}`")
            lines.append("")
            lines.append(f"- File: `{details['file']}`")
            for name, exists in details["exports"].items():
                lines.append(f"- `{name}`: {'OK' if exists else 'MISSING'}")
            lines.append("")

    lines.extend(["## Final submission manifest", ""])
    for item in report["manifest"]:
        lines.append(
            f"- `{item['path']}` "
            f"({item['size_bytes']} bytes, {item['category']})"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and audit a C3 submission directory."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Submission root; defaults to current directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicates, temp patches and caches.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the full unittest discovery run.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / "scheduler").is_dir():
        raise SystemExit(
            f"Not a submission root: missing {root / 'scheduler'}"
        )

    cleanup_candidates = find_cleanup_candidates(root)
    cleanup_actions = clean_candidates(
        root,
        cleanup_candidates,
        apply=args.apply,
    )

    findings: list[Finding] = []
    findings.extend(syntax_check(root))

    import_findings, import_details = import_path_check(root)
    findings.extend(import_findings)

    init_findings, init_details = inspect_scheduler_init(root)
    findings.extend(init_findings)

    findings.extend(source_leak_check(root))

    test_suite = (
        {"passed": True, "results": [], "skipped": True}
        if args.skip_tests
        else run_test_suite(root)
    )

    if not test_suite["passed"]:
        findings.append(
            Finding(
                "error",
                "tests",
                "tests/",
                "Full unittest discovery failed",
            )
        )

    manifest = generate_manifest(root)

    error_count = sum(item.severity == "error" for item in findings)
    warning_count = sum(item.severity == "warning" for item in findings)

    report = {
        "root": str(root),
        "mode": "apply" if args.apply else "dry-run",
        "cleanup": {
            "candidates": [rel(path, root) for path in cleanup_candidates],
            "actions": cleanup_actions,
        },
        "findings": [asdict(item) for item in findings],
        "imports": import_details,
        "scheduler_init": init_details,
        "test_suite": test_suite,
        "manifest": manifest,
        "summary": {
            "passed": error_count == 0 and test_suite["passed"],
            "errors": error_count,
            "warnings": warning_count,
        },
    }

    json_path = root / "final_submission_audit.json"
    md_path = root / "FINAL_SUBMISSION_MANIFEST.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )

    print("=" * 72)
    print("C3 FINAL SUBMISSION AUDIT")
    print("=" * 72)
    print(f"mode: {report['mode']}")
    print(f"cleanup candidates: {len(cleanup_candidates)}")
    print(f"errors: {error_count}")
    print(f"warnings: {warning_count}")
    print(f"test suite passed: {test_suite['passed']}")
    print(f"overall passed: {report['summary']['passed']}")
    print(f"\nJSON report: {json_path}")
    print(f"Manifest: {md_path}")

    if error_count or not test_suite["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
