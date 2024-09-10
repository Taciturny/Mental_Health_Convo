import re
import uuid

import pandas as pd


class DataPreprocessor:
    def remove_irrelevant_content(self, text):
        """
        Removes irrelevant content from the given text, including emails, phone numbers, URLs, social media handles,
        dates, times, addresses, ZIP codes, promotional phrases, legal disclaimers, special characters, hashtags, HTML tags,
        Markdown links, specific phrases, and certain symbols.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Define patterns to remove
        patterns = [
            r"[\w\.-]+@[\w\.-]+",  # Emails
            r"\+?\d[\d\-\s\(\)]{9,}\d",  # Phone numbers
            r"https?://(?:www\.)?[\w/\-?=%.]+\.[\w/\-?=%.]+",  # URLs
            r"@[A-Za-z0-9_.]+",  # Social media handles
            r"\b(?:\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}|\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2})\b",  # Dates
            r"\b(?:\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?)\b",  # Times
            r"\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Way|Drive|Dr)\b",  # Addresses
            r"\b\d{5}(?:-\d{4})?\b",  # ZIP codes
            r"(limited time offer|buy now|subscribe for more|click here to learn more|order today)",  # Promotional phrases
            r"(Terms and conditions apply|All rights reserved|Privacy policy)",  # Legal disclaimers
            r"[™©®]",  # Special characters
            r"\#[A-Za-z0-9_]+",  # Hashtags
            r"<[^>]+>",  # HTML tags
            r"\[[^\]]+\]\([^\)]+\)",  # Markdown links
            r"\(If you would prefer to contact me on my personal blog, please fill out the form below to stay updated\)",  # Specific phrase
            r"[â€™]",  # Specific symbols like "Iâ€™"
        ]

        # Remove all patterns
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_and_clean_text(self, df):
        """
        Normalizes and cleans text in the 'input' and 'output' columns of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.

        Returns:
            pd.DataFrame: The DataFrame with cleaned text.
        """
        for col in ["input", "output"]:
            df[col] = df[col].apply(
                lambda text: self.remove_irrelevant_content(
                    text.lower().strip()
                )
            )
        return df

    def load_parquet_data(self, input_parquet_path):
        """
        Loads data from a Parquet file.

        Args:
            input_parquet_path (str): The path to the Parquet file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        return pd.read_parquet(input_parquet_path)

    def load_csv_data(self, input_csv_path):
        """
        Loads data from a CSV file and renames columns to match the expected format.

        Args:
            input_csv_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded and modified DataFrame.
        """
        df = pd.read_csv(input_csv_path)
        # Rename columns to match the Parquet data format
        df = df.rename(columns={"Questions": "question", "Answers": "answer"})
        return df

    def preprocess_data(self, df):
        """
        Preprocesses the DataFrame by removing duplicates, normalizing and cleaning text, and adding a unique identifier.

        Args:
            df (pd.DataFrame): The DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        # Remove duplicates
        df = df.drop_duplicates()

        # Normalize and clean text
        df = self.normalize_and_clean_text(df)

        # Add a unique document identifier
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        return df

    def save_data(self, df, output_path):
        """
        Saves the preprocessed DataFrame to a Parquet file.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.
            output_path (str): The path to the output Parquet file.
        """
        # Rename columns
        processed_data = df.rename(
            columns={"question": "question", "answer": "answer"}
        )
        columns_order = ["id", "question", "answer"]
        processed_data = processed_data[columns_order]
        processed_data.to_parquet(output_path)
        print(
            f"Data saved to {output_path} with columns: {', '.join(processed_data.columns)}"
        )


def main(parquet_path, csv_path, output_path):
    """
    Main function to preprocess data from Parquet and CSV files and save the combined result.

    Args:
        parquet_path (str): The path to the Parquet file.
        csv_path (str): The path to the CSV file.
        output_path (str): The path to save the preprocessed Parquet file.
    """
    preprocessor = DataPreprocessor()

    # Load both datasets
    df_parquet = preprocessor.load_parquet_data(parquet_path)
    df_csv = preprocessor.load_csv_data(csv_path)

    print(f"Shape of original Parquet data: {df_parquet.shape}")
    print(f"Shape of original CSV data: {df_csv.shape}")

    # Preprocess both datasets
    df_parquet_processed = preprocessor.preprocess_data(df_parquet)
    df_csv_processed = preprocessor.preprocess_data(
        df_csv.drop(columns=["Question_ID"], errors="ignore")
    )

    # Concatenate the two dataframes
    combined_df = pd.concat(
        [df_parquet_processed, df_csv_processed], ignore_index=True
    )

    print(f"Shape of combined data: {combined_df.shape}")

    # Print some statistics
    print(f"Number of unique inputs: {combined_df['question'].nunique()}")
    print(
        f"Average responses per input: {combined_df.groupby('question').size().mean():.2f}"
    )

    # Save the combined data
    preprocessor.save_data(combined_df, output_path)


if __name__ == "__main__":
    parquet_path = "data/hea.parquet"
    csv_path = "data/Mental_Health_FAQ.csv"
    output_path = "data/preprocessed_data.parquet"
    main(parquet_path, csv_path, output_path)
