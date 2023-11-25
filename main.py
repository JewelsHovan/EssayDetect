from Generators.GPTGenerator import LLMGenerator
import pandas as pd

if __name__ == "__main__":
    generator = LLMGenerator()
    n = 1  # Number of iterations to generate essays
    train_prompts_path = 'LLM Detect AI Generated Text/train_prompts.csv'
    prompts_df = pd.read_csv(train_prompts_path)
    df_generated = generator.generate_essays(n, prompts_df, use_palm=True)
    print(df_generated.head())
    df_generated.to_csv('PaLM_generated_essays.csv', index=False)