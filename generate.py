import openai
from openai import OpenAI
import pandas as pd
import uuid
import os
import re

def generate_essay(prompt, source_text):
    """
    Generates an essay based on a given prompt and source text using the Chat Completions API.

    :param prompt: The prompt for the essay.
    :param source_text: The source text related to the prompt.
    """
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=400,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"{prompt}\n\n{source_text}\n\nWrite an essay about the above without titles or formal sign-offs:"}
        ]
    )

    # Accessing the generated text from the response
    generated_response = response.choices[0].message.content.strip()
    clean_text = re.sub(r'\n+', '\n', generated_response)  # Remove extra new line characters
    clean_text = re.sub(r'(Dear .*?:|Sincerely,.*|Yours truly,.*|\[Your Name\])', '', clean_text, flags=re.IGNORECASE)  # Remove formal greetings and sign-offs

    return clean_text.strip()


def main(n, prompts_df):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    generated_essays = []
    for _ in range(n):
        for _, row in prompts_df.iterrows():
            prompt_id = row['prompt_id']
            prompt_name = row['prompt_name']
            prompt_instructions = row['instructions']
            source_text = row['source_text']

            generated_text = generate_essay(prompt_instructions, source_text)
            essay_id = str(uuid.uuid4())[:8]  # Generate a unique ID for each essay
            generated_essays.append([essay_id, prompt_id, generated_text, 1])

    generated_df = pd.DataFrame(generated_essays, columns=['id', 'prompt_id', 'text', 'generated'])
    return generated_df

if __name__ == "__main__":
    n = 1  # Number of iterations to generate essays
    train_prompts_path = 'LLM Detect AI Generated Text/train_prompts.csv'
    prompts_df = pd.read_csv(train_prompts_path)

    df_generated = main(n, prompts_df)
    print(df_generated.head())
    df_generated.to_csv('GPT_generated_essays.csv', index=False)
