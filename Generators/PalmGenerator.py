import pandas as pd
import google.generativeai as palm
import uuid
import re
import os

class PalmLLM:
    def __init__(self, api_key=None):
        self.api_key = api_key or 'YOUR_PALM_API_KEY'
        palm.configure(api_key=self.api_key)

    def generate_text(self, prompt, source_text):
        """
        Generates text based on a given prompt and source text using Google's PaLM model.

        :param prompt: The prompt for the text generation.
        :param source_text: The source text related to the prompt.
        """
        response = palm.chat(
            model='models/chat-bison-001',  # Replace with the specific PaLM model you intend to use
            messages= f"{prompt}\n\n{source_text} No title headers or bullet points, only paragraphs of text",
            temperature=1,
            candidate_count=1
        )

        # Accessing the generated text from the response
        generated_response = response.candidates[0]['content']
        clean_text = re.sub(r'\n+', '\n', generated_response)  # Remove extra new lines
        # Additional cleaning or processing can be added here

        return clean_text

    def generate_texts(self, n, prompts_df):
        """
        Generates multiple texts using the PaLM model.

        :param n: Number of iterations to generate texts.
        :param prompts_df: DataFrame containing prompts and source texts.
        """
        generated_texts = []
        for _ in range(n):
            for _, row in prompts_df.iterrows():
                prompt_id = row['prompt_id']
                source_text = row['source_text']

                generated_text = self.generate_text(row['instructions'], source_text)
                text_id = str(uuid.uuid4())[:8]  # Generate a unique ID for each text
                generated_texts.append([text_id, prompt_id, generated_text, 1])

        generated_df = pd.DataFrame(generated_texts, columns=['id', 'prompt_id', 'text', 'generated'])
        return generated_df
