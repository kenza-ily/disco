#!/usr/bin/env python3
"""
Fix path handling in consolidated notebooks by replacing broken Path(...) patterns
with NOTEBOOK_DIR references.
"""

import json
from pathlib import Path
import re


def fix_notebook_paths(notebook_path: Path, notebook_name: str):
    """Fix all path references in a notebook."""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = False

    # Pattern to match: Path(str(Path('parsing.ipynb').resolve())).parent
    pattern = rf"Path\(str\(Path\('{notebook_name}'\)\.resolve\(\)\)\)\.parent"

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # Replace the broken pattern with NOTEBOOK_DIR
                    new_line = re.sub(pattern, 'NOTEBOOK_DIR', line)

                    # Also fix LaTeX generation paths (str(Path('parsing.ipynb').resolve()))
                    latex_pattern = rf"str\(Path\('{notebook_name}'\)\.resolve\(\)\)"
                    new_line = re.sub(latex_pattern, f"str(NOTEBOOK_DIR / '{notebook_name}')", new_line)

                    if new_line != line:
                        modified = True
                        print(f"  Fixed: {line.strip()[:80]}...")

                    new_source.append(new_line)

                cell['source'] = new_source

    if modified:
        # Save the fixed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✅ Fixed {notebook_path.name}")
        return True
    else:
        print(f"⚠️  No changes needed for {notebook_path.name}")
        return False


def main():
    """Fix all three consolidated notebooks."""
    notebooks_dir = Path(__file__).parent

    notebooks = [
        ('parsing.ipynb', 'parsing.ipynb'),
        ('qa.ipynb', 'qa.ipynb'),
        ('ocr_vs_vlm.ipynb', 'ocr_vs_vlm.ipynb'),
    ]

    print("Fixing notebook path references...\n")

    total_fixed = 0
    for filename, notebook_name in notebooks:
        notebook_path = notebooks_dir / filename
        if notebook_path.exists():
            print(f"\nProcessing {filename}:")
            if fix_notebook_paths(notebook_path, notebook_name):
                total_fixed += 1
        else:
            print(f"❌ Not found: {filename}")

    print(f"\n✅ Fixed {total_fixed} notebooks")


if __name__ == '__main__':
    main()
