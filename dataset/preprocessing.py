import pandas as pd
from src.utils import BPETokenizer, Config

class Preprocess:
    def __init__(self, config) -> None:
        self.dataset = None 
        self.tokenizer = BPETokenizer(vocab_size=config.vocab_size)  # Initialize BPE Tokenizer

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from an Excel file."""

        return pd.read_excel(filepath)

    def apply_preprocess(self, filepath: str) -> None:
        """Preprocess the dataset by loading and cleaning it."""
        
        data = self.load_data(filepath)
        data.dropna(inplace=True)
        self.dataset = ' '.join(data['telugu_lyrics'].astype(str))

    def save_dataset(self, filename: str) -> None:
        """Save the processed dataset to a text file."""

        with open(filename + '.txt', 'w') as f:
            f.write(self.dataset)

    def train_tokenizer(self, texts: list) -> None:
        """Train the BPE tokenizer on the dataset."""

        self.tokenizer.fit(texts, verbose=True)
        self.tokenizer.save_vocab('bpe_tokenizer')  # Save the tokenizer as bpe_tokenizer.json

    def process(self, input_filepath: str, output_filename: str) -> None:
        """Main method to process the dataset and train the tokenizer."""

        self.apply_preprocess(input_filepath)
        self.save_dataset(output_filename)
        self.train_tokenizer(self.dataset.split())  # Train tokenizer on the dataset


if __name__ == "__main__":
    preprocessor = Preprocess(Config())
    preprocessor.process('./dataset/telugu_lyrics.xlsx', 'lyrics')

