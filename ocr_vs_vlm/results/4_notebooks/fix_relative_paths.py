#!/usr/bin/env python3
"""Fix relative paths in notebook configs (../../ -> ../)"""

import json
from pathlib import Path


def fix_relative_paths_in_notebook(notebook_path: Path):
    """Replace ../../ with ../ in notebook cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # Replace ../../2_clean with ../2_clean
                    new_line = line.replace("'../../2_clean", "'../2_clean")
                    new_line = new_line.replace('"../../2_clean', '"../2_clean')
                    # Replace ../../3_embeddings with ../3_embeddings
                    new_line = new_line.replace("'../../3_embeddings", "'../3_embeddings")
                    new_line = new_line.replace('"../../3_embeddings', '"../3_embeddings')

                    if new_line != line:
                        modified = True
                        print(f"  Fixed: {line.strip()[:100]}")

                    new_source.append(new_line)
                cell['source'] = new_source

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✅ Fixed {notebook_path.name}")
        return True
    else:
        print(f"⚠️  No path changes needed for {notebook_path.name}")
        return False


def main():
    notebooks_dir = Path(__file__).parent
    notebooks = ['parsing.ipynb', 'qa.ipynb', 'ocr_vs_vlm.ipynb']

    print("Fixing relative paths in notebooks...\n")
    for nb in notebooks:
        nb_path = notebooks_dir / nb
        if nb_path.exists():
            print(f"\nProcessing {nb}:")
            fix_relative_paths_in_notebook(nb_path)
        else:
            print(f"❌ Not found: {nb}")


if __name__ == '__main__':
    main()
