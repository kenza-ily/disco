#!/usr/bin/env python3
"""
Test script for Mistral OCR API
"""
import base64
import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv(Path(".env.local"))

# Get a test image
test_image_path = Path("ocr_vs_vlm/datasets_subsets/icdar_mini/icdar_mini_Latin.json")
# Actually let's find a real image
test_images = list(Path("datasets/parsing/ICDAR/ImagesPart1").glob("*.jpg"))
if test_images:
    test_image_path = test_images[0]
    print(f"Using test image: {test_image_path}")
else:
    print("No test images found!")
    exit(1)

# Load and encode image
with open(test_image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

print(f"Endpoint: {endpoint}")
print(f"API Key: {api_key[:20]}...")

endpoint = endpoint.rstrip("/")
endpoint_url = f"{endpoint}/providers/mistral/azure/ocr"

print(f"Full URL: {endpoint_url}")
print()

# Test the correct format: document_url with data: scheme
document_url = f"data:image/jpeg;base64,{image_data}"

headers = {
    "Content-Type": "application/json",
    "api-key": api_key.strip(),
}

payload = {
    "model": "mistral-document-ai-2505",
    "document": {
        "type": "document_url",
        "document_url": document_url
    }
}

print(f"Headers: {headers}")
print(f"Payload keys: {payload.keys()}")
print(f"Document type: {payload['document']['type']}")
print()

try:
    print("Sending request to Mistral OCR API...")
    response = requests.post(
        endpoint_url,
        headers=headers,
        json=payload,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    
    try:
        resp_json = response.json()
        if response.status_code == 200:
            print("✓ SUCCESS!")
            print(f"\nResponse structure:")
            print(f"  - keys: {list(resp_json.keys())}")
            
            if "pages" in resp_json:
                print(f"  - pages count: {len(resp_json['pages'])}")
                if resp_json['pages']:
                    page = resp_json['pages'][0]
                    print(f"  - page keys: {list(page.keys())}")
                    
                    if 'text' in page:
                        extracted_text = page['text']
                        print(f"\n✓ Extracted text ({len(extracted_text)} chars):")
                        print(extracted_text[:500])
                    else:
                        print(f"\nPage content: {json.dumps(page, indent=2)[:500]}")
        else:
            print("✗ Error response:")
            print(json.dumps(resp_json, indent=2)[:1000])
    except Exception as e2:
        print(f"Error parsing response: {e2}")
        print(f"Response Body (raw): {response.text[:500]}")
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test complete")

