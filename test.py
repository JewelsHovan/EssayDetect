import torch
from transformers import pipeline

def main():
    # Initialize the Zephyr-7B model pipeline
    # Use a pipeline as a high-level helper
    zephyr_pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
    # Sample conversation messages
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    ]

    # Format the messages using the tokenizer's chat template
    prompt = zephyr_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate a response from the model
    outputs = zephyr_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    # Print the generated text
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
