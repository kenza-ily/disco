#!/usr/bin/env python3
"""
Simple script to call an LLM and save the output to output.md
"""

import sys
import os

# Add parent directory to path to import llm_settings
sys.path.append('..')

from llm_settings import get_azure_openai_client

def main():
    # Get the Azure OpenAI client
    client = get_azure_openai_client()

    # Make a simple LLM call
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you tell me a fun fact about AI?"}
        ],
        max_tokens=150
    )

    # Get the response content
    output = response.choices[0].message.content

    # Save to output.md in the output subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'output', 'output.md')
    with open(output_path, 'w') as f:
        f.write("# LLM Output\n\n")
        f.write(output)
        f.write("\n")

    print(f"Output saved to {output_path}")
    print(f"Response: {output}")

if __name__ == "__main__":
    main()