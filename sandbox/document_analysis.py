#!/usr/bin/env python3
"""
Script to analyze a PDF document using OCR or VLM models
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add parent directory to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from sandbox/
sys.path.insert(0, project_root)

from ocr_vs_vlm.api_calls import call_ocr, call_vlm


def analyze_document_ocr(pdf_path: str, model: str = "azure_intelligence"):
    """Analyze a PDF document using OCR models."""

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} does not exist")
        return

    print(f"Using OCR model: {model}")
    documents = call_ocr(pdf_path, model=model)
    
    # Extract text content
    output = {
        "filename": Path(pdf_path).name,
        "model_type": "ocr",
        "model": model,
        "pages": []
    }
    
    # Process each document (page)
    for doc in documents:
        content = doc.page_content
        metadata = doc.metadata
        
        page_data = {
            "page_number": metadata.get("page_number", 1),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "unit": metadata.get("unit", "inch"),
            "lines": [{"text": line.strip(), "bounding_box": None} for line in content.split('\n') if line.strip()]
        }
        
        output["pages"].append(page_data)

    return output


def analyze_document_vlm(pdf_path: str, model: str = "donut"):
    """Analyze a PDF document using VLM models."""

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} does not exist")
        return

    print(f"Using VLM model: {model}")
    query = "Extract all text and structure from this document"
    result = call_vlm(pdf_path, model=model, query=query)
    
    output = {
        "filename": Path(pdf_path).name,
        "model_type": "vlm",
        "model": model,
        "result": result
    }
    
    return output


def save_output(output: dict, pdf_path: str):
    """Save analysis output to files with model name in filename."""
    
    filename_stem = Path(pdf_path).stem
    model_name = output.get("model", "unknown").replace("_", "-")
    
    # Save JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_filename = f"{filename_stem}_{model_name}.json"
    json_file = os.path.join(script_dir, 'output', json_filename)
    
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis saved to {json_file}")
    
    # Save plain text if OCR
    if output.get("model_type") == "ocr":
        text_content = ""
        for page in output.get("pages", []):
            text_content += f"\n--- Page {page['page_number']} ---\n"
            for line in page.get("lines", []):
                text_content += line["text"] + "\n"
        
        text_filename = f"{filename_stem}_{model_name}.txt"
        text_file = os.path.join(script_dir, 'output', text_filename)
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"Plain text saved to {text_file}")
    
    # Save VLM result as text
    elif output.get("model_type") == "vlm":
        text_filename = f"{filename_stem}_{model_name}.txt"
        text_file = os.path.join(script_dir, 'output', text_filename)
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(output.get("result", ""))
        
        print(f"VLM output saved to {text_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a document using OCR or VLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python document_analysis.py datasets/document.pdf --type ocr --model azure_intelligence
  python document_analysis.py datasets/document.pdf --type vlm --model donut
  python document_analysis.py datasets/document.pdf --type ocr --model mistral_ocr
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF or image file to analyze"
    )
    parser.add_argument(
        "--type",
        choices=["ocr", "vlm"],
        required=True,
        help="Type of analysis: OCR or Vision Language Model"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="""Model to use. 
                OCR models: azure_intelligence, mistral_ocr, mistral_ocr_3, donut, deepseek_ocr
                VLM models: gpt5_mini, gpt5_nano, claude_sonnet, claude_haiku, qwen_vl"""
    )
    
    args = parser.parse_args()
    
    # If it's a relative path, make it relative to the project root
    pdf_path_arg = args.pdf_path
    if not os.path.isabs(pdf_path_arg):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level from sandbox/
        pdf_path = os.path.join(project_root, pdf_path_arg)
    else:
        pdf_path = pdf_path_arg

    print(f"Analyzing document: {pdf_path}")
    print(f"Type: {args.type}, Model: {args.model}")
    
    try:
        if args.type == "ocr":
            output = analyze_document_ocr(pdf_path, model=args.model)
            if output:
                save_output(output, pdf_path)
                # Print summary
                print(f"\nDocument: {output['filename']}")
                print(f"Pages analyzed: {len(output.get('pages', []))}")
                print(f"Total lines: {sum(len(page.get('lines', [])) for page in output.get('pages', []))}")
        
        elif args.type == "vlm":
            output = analyze_document_vlm(pdf_path, model=args.model)
            if output:
                save_output(output, pdf_path)
                # Print summary
                print(f"\nDocument: {output['filename']}")
                print(f"Model: {output['model']}")
                print(f"Result length: {len(output.get('result', ''))} characters")
    
    except Exception as e:
        print(f"Error analyzing document: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()