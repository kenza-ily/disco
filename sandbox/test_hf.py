import sys
sys.path.append('..')

from llms.huggingface_client import generate_text

if __name__ == "__main__":
    prompt = "Hello, how are you today?"
    print(f"Prompt: {prompt}")
    response = generate_text(prompt)
    print(f"Response: {response}")