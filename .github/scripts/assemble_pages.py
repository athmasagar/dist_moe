# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Assemble the main Pages site and an isolated branch-review preview."""

import argparse
import shutil
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class LocalAssetParser(HTMLParser):
    """Check that HTML assets work beneath a GitHub Pages project path."""

    def __init__(self, root: Path) -> None:
        """Initialize the validator for one static-site directory.

        Args:
            root: Directory containing the site's index.html.
        """
        super().__init__()
        self.root = root.resolve()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Validate local script, stylesheet, favicon, and image references.

        Args:
            tag: HTML tag name.
            attrs: Attributes supplied by the HTML parser.
        """
        key = "href" if tag == "link" else "src"
        if tag not in {"script", "link", "img"}:
            return
        reference = dict(attrs).get(key)
        if not reference:
            return
        url = urlsplit(reference)
        if url.scheme or url.netloc:
            return
        if url.path.startswith("/"):
            raise ValueError(f"Asset must use a project-relative path: {reference}")
        target = (self.root / unquote(url.path)).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            raise ValueError(f"Missing or out-of-site asset: {reference}")


def copy_site(source: Path | None, destination: Path) -> bool:
    """Copy and validate an available static site.

    Args:
        source: Optional site directory; missing directories are omitted.
        destination: Empty destination directory for the copied site.

    Returns:
        Whether a complete site was copied.
    """
    if source is None or not source.is_dir():
        return False
    if not (source / "index.html").is_file():
        raise ValueError(f"Site is missing index.html: {source}")
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"Site contains a symlink: {source}")
    LocalAssetParser(source).feed((source / "index.html").read_text())
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return True


def assemble(production: Path | None, preview: Path | None, output: Path) -> None:
    """Build a fresh Pages artifact without replacing production with a preview.

    Args:
        production: Main-branch site, when available.
        preview: Review-branch site, when available.
        output: New directory to create for the deployment artifact.
    """
    output.mkdir(parents=True, exist_ok=False)
    has_production = copy_site(production, output)
    preview_path = Path("previews/visual-explainer")
    has_preview = copy_site(preview, output / preview_path)
    if not has_production and not has_preview:
        raise ValueError("Neither the main site nor the review preview is available")
    if not has_production:
        target = escape(preview_path.as_posix() + "/", quote=True)
        (output / "index.html").write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="refresh" content="0;url={target}">'
            '<title>Dist-MoE illustrated guide preview</title></head><body>'
            f'<a href="{target}">Open the illustrated guide preview</a>'
            "</body></html>\n"
        )
    (output / ".nojekyll").touch()
    print(f"Pages artifact: {output}")
    print(f"Main site included: {has_production}")
    print(f"Review preview included: {has_preview}")


def main() -> None:
    """Parse deployment paths and assemble the static Pages artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.production, args.preview, args.output)


if __name__ == "__main__":
    main()
