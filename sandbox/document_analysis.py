#!/usr/bin/env python3
"""
Script to analyze a PDF document using Azure Document Intelligence
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path to import llms
sys.path.append('..')

from llms.azure_client import get_azure_document_intelligence_client

def analyze_document(pdf_path: str):
    """Analyze a PDF document using Azure Document Intelligence."""

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} does not exist")
        return

    # Get the client
    client = get_azure_document_intelligence_client()

    # Open and analyze the document
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",  # Use the read model for general document analysis
            body=f
        )

    # Wait for the operation to complete
    result = poller.result()

    # Extract text content
    output = {
        "filename": Path(pdf_path).name,
        "model": "prebuilt-read",
        "pages": []
    }

    for page in result.pages:
        page_data = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "unit": page.unit,
            "lines": []
        }

        for line in page.lines:
            line_data = {
                "text": line.content,
                "bounding_box": line.polygon
            }
            page_data["lines"].append(line_data)

        output["pages"].append(page_data)

    # Save to JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'output', 'document_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_file}")

    # Also save plain text
    text_content = ""
    for page in output["pages"]:
        text_content += f"\n--- Page {page['page_number']} ---\n"
        for line in page["lines"]:
            text_content += line["text"] + "\n"

    text_file = os.path.join(script_dir, 'output', 'document_text.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_content)

    print(f"Plain text saved to {text_file}")

    # Print summary
    print(f"\nDocument: {output['filename']}")
    print(f"Pages analyzed: {len(output['pages'])}")
    print(f"Total lines: {sum(len(page['lines']) for page in output['pages'])}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python document_analysis.py <pdf_path>")
        print("Example: python document_analysis.py datasets/Generative\\ AI\\ in\\ Healthcare\\ eBook.pdf")
        sys.exit(1)
    
    # Get PDF path from command line argument
    pdf_path_arg = sys.argv[1]
    
    # If it's a relative path, make it relative to the project root
    if not os.path.isabs(pdf_path_arg):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from sandbox/
        pdf_path = os.path.join(project_root, pdf_path_arg)
    else:
        pdf_path = pdf_path_arg

    print(f"Analyzing document: {pdf_path}")
    analyze_document(pdf_path)

if __name__ == "__main__":
    main()