from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

# =========================
# Configuration
# =========================

ROOT_DIR = Path(r"C:\fingerprint-research\apps\api")
OUTPUT_FILE = Path(r"C:\fingerprint-research\apps\api\api_code_for_ai.txt")

# סיומות שנחשבות "קוד/טקסט/קונפיג" ומתאימות לייצוא ל-AI
INCLUDED_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx",
    ".css", ".scss", ".sass", ".less",
    ".html", ".htm",
    ".json", ".jsonc",
    ".md", ".mdx", ".txt",
    ".yml", ".yaml",
    ".xml",
    ".py",
    ".sh", ".bash",
    ".env", ".env.example",
    ".graphql", ".gql",
    ".svg",
    ".cjs", ".mjs",
    ".ini", ".cfg", ".conf",
    ".sql",
    ".prisma",
}

# שמות קבצים בלי סיומת שעדיין כדאי לכלול
INCLUDED_FILENAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    ".eslintignore",
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".prettierignore",
    ".prettierrc",
    ".prettierrc.js",
    ".prettierrc.cjs",
    ".prettierrc.json",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "tsconfig.json",
    "tsconfig.base.json",
    "nest-cli.json",
    "README.md",
}

# קבצים כבדים מאוד בדרך כלל לא מועילים לסוכן AI
MAX_FILE_SIZE_BYTES = 1_500_000  # 1.5MB

# אנקודינגים לנסות
ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


# =========================
# Helpers
# =========================

def should_include_file(path: Path) -> bool:
    """קובע האם הקובץ מתאים לייצוא."""
    if not path.is_file():
        return False

    try:
        size = path.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            return False
    except OSError:
        return False

    name = path.name
    suffix = path.suffix.lower()

    if name in INCLUDED_FILENAMES:
        return True

    if suffix in INCLUDED_EXTENSIONS:
        return True

    return False


def read_text_file(path: Path) -> str:
    """קורא קובץ טקסט עם fallback בין אנקודינגים."""
    for encoding in ENCODINGS_TO_TRY:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as e:
            return f"<<ERROR READING FILE: {e}>>"

    return "<<ERROR READING FILE: unsupported encoding or binary content>>"


def add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return "0001: "
    width = max(4, len(str(len(lines))))
    return "\n".join(f"{i:0{width}d}: {line}" for i, line in enumerate(lines, start=1))


def build_directory_tree(root: Path, included_files: list[Path]) -> str:
    """
    בונה עץ תיקיות רק עבור הקבצים שנכנסו לייצוא.
    """
    included_rel_paths = sorted(p.relative_to(root) for p in included_files)

    tree_lines = [root.name + "/"]
    all_nodes = set()

    for rel_path in included_rel_paths:
        current = Path()
        for part in rel_path.parts:
            current = current / part
            all_nodes.add(current)

    def has_children(node: Path) -> bool:
        return any(n.parent == node for n in all_nodes)

    def walk(prefix_path: Path, indent: str = ""):
        children = sorted(
            [node for node in all_nodes if node.parent == prefix_path],
            key=lambda p: [x.lower() for x in p.parts]
        )

        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            name = child.name + ("/" if has_children(child) else "")
            tree_lines.append(indent + connector + name)
            next_indent = indent + ("    " if is_last else "│   ")
            if has_children(child):
                walk(child, next_indent)

    walk(Path())
    return "\n".join(tree_lines)


def collect_files(root: Path) -> list[Path]:
    files = []
    for current_root, _, filenames in os.walk(root):
        current_path = Path(current_root)

        for filename in filenames:
            file_path = current_path / filename
            if should_include_file(file_path):
                files.append(file_path)

    files.sort(key=lambda p: str(p.relative_to(root)).lower())
    return files


def export_codebase(root: Path, output_file: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist or is not a directory: {root}")

    files = collect_files(root)
    tree = build_directory_tree(root, files)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# AI-Friendly Code Export\n\n")
        out.write(f"Exported at: {datetime.now().isoformat()}\n")
        out.write(f"Root: {root}\n")
        out.write(f"Output: {output_file}\n")
        out.write(f"Total files included: {len(files)}\n\n")

        out.write("## Directory Tree\n\n")
        out.write(tree)
        out.write("\n\n")

        out.write("## File Index\n\n")
        for i, file_path in enumerate(files, start=1):
            rel_path = file_path.relative_to(root)
            out.write(f"{i:04d}. {rel_path.as_posix()}\n")
        out.write("\n\n")

        out.write("## File Contents\n\n")

        for i, file_path in enumerate(files, start=1):
            rel_path = file_path.relative_to(root)
            content = read_text_file(file_path)
            numbered_content = add_line_numbers(content)

            out.write("=" * 100 + "\n")
            out.write(f"FILE #{i:04d}\n")
            out.write(f"PATH: {rel_path.as_posix()}\n")
            out.write(f"SIZE_BYTES: {file_path.stat().st_size}\n")
            out.write(f"EXTENSION: {file_path.suffix or '[no extension]'}\n")
            out.write("-" * 100 + "\n")
            out.write(numbered_content)
            out.write("\n\n")

    print(f"Done. Exported {len(files)} files to:\n{output_file}")


if __name__ == "__main__":
    export_codebase(ROOT_DIR, OUTPUT_FILE)