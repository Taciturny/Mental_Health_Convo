# import pandas as pd
# import numpy as np

# def load_data(input_parquet_path):
#     """Load data from a Parquet file."""
#     return pd.read_parquet(input_parquet_path)

# def remove_duplicates(df):
#     """Remove duplicate rows from the DataFrame."""
#     return df.drop_duplicates()

# def normalize_text(df):
#     """Normalize text in the 'Context' and 'Response' columns."""
#     df['Context'] = df['Context'].str.lower().str.replace('[^\w\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True)
#     df['Response'] = df['Response'].str.lower().str.replace('[^\w\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True)
#     return df

# def preprocess_data(df):
#     """Preprocess the DataFrame."""
#     # Remove duplicates
#     df_no_duplicates = remove_duplicates(df)
    
#     # Group by 'Context' and aggregate 'Response'
#     df_grouped = df_no_duplicates.groupby('Context')['Response'].agg(list).reset_index()
    
#     # Filter to keep only contexts with multiple answers
#     df_multiple_answers = df_grouped[df_grouped['Response'].map(len) > 1]
    
#     # Filter to keep only contexts with a single answer
#     df_single_answers = df_grouped[df_grouped['Response'].map(len) == 1]
#     df_single_answers['Response'] = df_single_answers['Response'].map(lambda x: [x[0]])
    
#     # Ensure 'Response' column is always a list
#     df_multiple_answers['Response'] = df_multiple_answers['Response'].apply(lambda x: x if isinstance(x, list) else [x])
#     df_single_answers['Response'] = df_single_answers['Response'].apply(lambda x: x if isinstance(x, list) else [x])
    
#     # Combine multiple and single answer dataframes
#     df_processed = pd.concat([df_multiple_answers, df_single_answers], ignore_index=True)
#     return df_processed

# def save_data(df, output_parquet_path):
#     """Save the DataFrame to a Parquet file."""
#     df.to_parquet(output_parquet_path, index=False)
#     print(f"Preprocessed data saved to {output_parquet_path}")

# def main(input_parquet_path, output_parquet_path):
#     """Main function to preprocess data."""
#     df = load_data(input_parquet_path)
#     df = normalize_text(df)
#     df_processed = preprocess_data(df)
#     save_data(df_processed, output_parquet_path)

# if __name__ == "__main__":
#     input_parquet_path = 'data/men-hea-data.parquet'
#     output_parquet_path = 'data/preprocessed_data.parquet'
#     main(input_parquet_path, output_parquet_path)

# import pandas as pd
# import uuid

# def load_data(input_parquet_path):
#     return pd.read_parquet(input_parquet_path)

# def preprocess_data(df):
#     # Normalize text
#     for col in ['Context', 'Response']:
#         df[col] = df[col].str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    
#     # Group by Context and aggregate unique Responses
#     df_grouped = df.groupby('Context')['Response'].agg(lambda x: ' | '.join(set(x))).reset_index()
    
#     # Add a unique document identifier
#     df_grouped['document'] = [str(uuid.uuid4()) for _ in range(len(df_grouped))]
    
#     return df_grouped

# def save_data(df, output_json_path):
#     # Convert to the desired format for Weaviate
#     weaviate_data = df.rename(columns={'Context': 'context', 'Response': 'response'}).to_dict('records')
    
#     # Save as JSON
#     import json
#     with open(output_json_path, 'w') as f:
#         json.dump(weaviate_data, f, indent=2)
#     print(f"Preprocessed data saved to {output_json_path}")

# def main(input_parquet_path, output_json_path):
#     df = load_data(input_parquet_path)
#     df_processed = preprocess_data(df)
#     print(f"Shape of preprocessed data: {df_processed.shape}")
#     save_data(df_processed, output_json_path)

# if __name__ == "__main__":
#     input_parquet_path = 'data/men-hea-data.parquet'
#     output_json_path = 'data/weaviate_input.json'
#     main(input_parquet_path, output_json_path)


# import pandas as pd
# import uuid

# def load_data(input_parquet_path):
#     return pd.read_parquet(input_parquet_path)

# def preprocess_data(df):
#     # Normalize text
#     for col in ['Context', 'Response']:
#         df[col] = df[col].str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    
#     # Add a unique document identifier
#     df['document'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
#     return df

# def save_data(df, output_json_path):
#     # Convert to the desired format for Weaviate
#     weaviate_data = df.rename(columns={'Context': 'context', 'Response': 'response'}).to_dict('records')
    
#     # Save as JSON
#     import json
#     with open(output_json_path, 'w') as f:
#         json.dump(weaviate_data, f, indent=2)
#     print(f"Preprocessed data saved to {output_json_path}")

# def main(input_parquet_path, output_json_path):
#     df = load_data(input_parquet_path)
#     df_processed = preprocess_data(df)
#     print(f"Shape of preprocessed data: {df_processed.shape}")
#     save_data(df_processed, output_json_path)

# if __name__ == "__main__":
#     input_parquet_path = 'data/men-hea-data.parquet'
#     output_json_path = 'data/new_weaviate_input.json'
#     main(input_parquet_path, output_json_path)


# import pandas as pd
# import uuid
# from textblob import TextBlob

# def load_data(input_parquet_path):
#     return pd.read_parquet(input_parquet_path)

# def preprocess_data(df):
#     # Normalize text
#     for col in ['Context', 'Response']:
#         df[col] = df[col].str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    
#     # Add sentiment scores
#     df['context_sentiment'] = df['Context'].apply(lambda x: TextBlob(x).sentiment.polarity)
#     df['response_sentiment'] = df['Response'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
#     # Add a unique document identifier
#     df['document'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
#     return df

# def save_data(df, output_json_path):
#     # Convert to the desired format for Weaviate
#     weaviate_data = df.rename(columns={'Context': 'context', 'Response': 'response'}).to_dict('records')
    
#     # Save as JSON
#     import json
#     with open(output_json_path, 'w') as f:
#         json.dump(weaviate_data, f, indent=2)
#     print(f"Preprocessed data saved to {output_json_path}")

# def main(input_parquet_path, output_json_path):
#     df = load_data(input_parquet_path)
#     print(f"Shape of original data: {df.shape}")
    
#     df_processed = preprocess_data(df)
#     print(f"Shape of preprocessed data: {df_processed.shape}")
    
#     # Print some statistics
#     print(f"Number of unique contexts: {df_processed['Context'].nunique()}")
#     print(f"Average responses per context: {df_processed.groupby('Context').size().mean():.2f}")
    
#     save_data(df_processed, output_json_path)

# if __name__ == "__main__":
#     input_parquet_path = 'data/men-hea-data.parquet'
#     output_json_path = 'data/new_weaviate_input.json'
#     main(input_parquet_path, output_json_path)


import pandas as pd
import uuid
from textblob import TextBlob

def load_data(input_parquet_path):
    return pd.read_parquet(input_parquet_path)

def preprocess_data(df):
    # Normalize text
    for col in ['Context', 'Response']:
        df[col] = df[col].str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    
    # Add sentiment scores
    df['context_sentiment'] = df['Context'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['response_sentiment'] = df['Response'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Add a unique document identifier
    df['document'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    return df

def save_data(df, output_json_path):
    # Convert to the desired format for Weaviate
    weaviate_data = df.rename(columns={'Context': 'context', 'Response': 'response'}).to_dict('records')
    
    # Save as JSON
    import json
    with open(output_json_path, 'w') as f:
        json.dump(weaviate_data, f, indent=2)
    print(f"Preprocessed data saved to {output_json_path}")

def main(input_parquet_path, output_json_path):
    df = load_data(input_parquet_path)
    print(f"Shape of original data: {df.shape}")
    
    df_processed = preprocess_data(df)
    print(f"Shape of preprocessed data: {df_processed.shape}")
    
    # Print some statistics
    print(f"Number of unique contexts: {df_processed['Context'].nunique()}")
    print(f"Average responses per context: {df_processed.groupby('Context').size().mean():.2f}")
    
    save_data(df_processed, output_json_path)

if __name__ == "__main__":
    input_parquet_path = 'data/men-hea-data.parquet'
    output_json_path = 'data/new_weaviate_input2.json'
    main(input_parquet_path, output_json_path)
