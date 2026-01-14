#!/usr/bin/env python3
"""
Simple script to call an LLM and save the output to output.md
Usage: python llm_call.py --client azure --model gpt-4 --prompt "Hello, tell me a fun fact about AI"
"""

import sys
import os
import argparse
import json

# Add parent directory to path to import llms
sys.path.append('..')

from llms.llm_settings import get_bedrock_client

try:
    from llms.llm_settings import get_langchain_azure_openai, get_langchain_huggingface_pipeline
    from llms.huggingface_client import langchain_generate_text
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Call an LLM with specified client and model")
    parser.add_argument("--client", choices=["azure", "hf", "bedrock"], required=True, help="Client to use: azure, hf, or bedrock")
    parser.add_argument("--model", required=True, help="Model name or address (for bedrock, use model ID like 'anthropic.claude-3-5-sonnet-20241022-v2:0')")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the LLM")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens for response (bedrock only)")
    parser.add_argument("--profile", help="AWS profile name for bedrock (e.g., 'eu-dev', 'default')")
    
    args = parser.parse_args()
    
    if args.client == "azure":
        if not LANGCHAIN_AVAILABLE:
            print("Error: LangChain not available for Azure client")
            sys.exit(1)
        # Get the LangChain Azure OpenAI client
        llm = get_langchain_azure_openai(args.model)
        
        # Make the LLM call
        response = llm.invoke(args.prompt)
        
        # Get the response content
        output = response.content
        
    elif args.client == "hf":
        if not LANGCHAIN_AVAILABLE:
            print("Error: LangChain not available for HuggingFace client")
            sys.exit(1)
        # Use LangChain Hugging Face
        output = langchain_generate_text(args.prompt, model_name=args.model)
    
    elif args.client == "bedrock":
        # Use AWS Bedrock for Claude models
        bedrock = get_bedrock_client(profile_name=args.profile)
        
        # Prepare request body for Claude models
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": args.max_tokens,
            "messages": [
                {"role": "user", "content": args.prompt}
            ]
        })
        
        # Invoke the model
        response = bedrock.invoke_model(
            modelId=args.model,
            contentType='application/json',
            accept='application/json',
            body=body
        )
        
        # Parse response
        result = json.loads(response['body'].read())
        output = result['content'][0]['text']
    
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