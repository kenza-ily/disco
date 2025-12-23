#!/usr/bin/env python3
"""
Simple script to call an LLM and save the output to output.md
Usage: python llm_call.py --client azure --model gpt-4 --prompt "Hello, tell me a fun fact about AI"
"""

import sys
import os
import argparse

# Add parent directory to path to import llms
sys.path.append('..')

from llms.llm_settings import get_langchain_azure_openai, get_langchain_huggingface_pipeline
from llms.huggingface_client import langchain_generate_text

def main():
    parser = argparse.ArgumentParser(description="Call an LLM with specified client and model")
    parser.add_argument("--client", choices=["azure", "hf"], required=True, help="Client to use: azure or hf")
    parser.add_argument("--model", required=True, help="Model name or address")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the LLM")
    
    args = parser.parse_args()
    
    if args.client == "azure":
        # Get the LangChain Azure OpenAI client
        llm = get_langchain_azure_openai(args.model)
        
        # Make the LLM call
        response = llm.invoke(args.prompt)
        
        # Get the response content
        output = response.content
        
    elif args.client == "hf":
        # Use LangChain Hugging Face
        output = langchain_generate_text(args.prompt, model_name=args.model)
    
    # Save to output.md in the output subdirectory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'output', 'output.md')
    with open(output_path, 'w') as f:
        f.write("# LLM Output\n\n")
        f.write(f"Client: {args.client}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Prompt: {args.prompt}\n\n")
        f.write("Response:\n")
        f.write(output)
        f.write("\n")

    print(f"Output saved to {output_path}")
    print(f"Response: {output}")

if __name__ == "__main__":
    main()