from pathlib import Path
import re 
from collections import Counter
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import Metaspace


ROOT_PATH = Path("/ghome/c5mcv07/C5_G7_MCV/Week_3")

def build_vocab(captions, min_word_freq=1):
    word_count = Counter()

    # Count frequency of each word in the captions
    for caption in captions:
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)  # Clean punctuation
        words = clean_caption.split()  # Split into words
        word_count.update(words)

    # Create the vocabulary, only including words that meet the frequency threshold
    vocab = {word: idx + len(SPECIAL_TOKENS) for idx, (word, count) in enumerate(word_count.items()) if count >= min_word_freq}

    # Add special tokens to the vocabulary
    for idx, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = idx

    return vocab

DATASET_PATH = ROOT_PATH / "dataset"
IMAGES_PATH = DATASET_PATH / "Food Images"
ANNOTATIONS_PATH = DATASET_PATH / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

OUTPUTS_PATH = ROOT_PATH / "outputs"

CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(CHARS)
IDX2CHAR = {k: v for k, v in enumerate(CHARS)}
CHAR2IDX = {v: k for k, v in enumerate(CHARS)}
TEXT_MAX_LEN = 201

SPECIAL_TOKENS = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']

annotations_df = pd.read_csv(ANNOTATIONS_PATH, index_col=0)
annotations_df = annotations_df[~annotations_df['Title'].apply(lambda x: isinstance(x, float))]
captions = annotations_df['Title'].tolist()
WORD_MAX_LEN = max(len(str(caption).split()) for caption in captions) + 5
WORD2IDX = build_vocab(captions)
IDX2WORD = {idx: word for word, idx in WORD2IDX.items()}
NUM_WORDS = len(WORD2IDX)
captions = [caption.replace("-", " ") for caption in captions]

text_file = OUTPUTS_PATH / "captions.txt"
with text_file.open("w", encoding="utf-8") as f:
    for caption in captions:
        f.write(caption + "\n")


# Initialize WordPiece tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Metaspace(replacement="▁")

# Train the tokenizer
trainer = trainers.WordPieceTrainer(
    vocab_size=500, 
    min_frequency=1,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train([str(text_file)], trainer)

# Save the trained tokenizer
tokenizer_path = OUTPUTS_PATH / "wordpiece-tokenizer.json"
tokenizer.save(str(tokenizer_path))

# Load the trained tokenizer
tokenizer = Tokenizer.from_file(str(tokenizer_path))

# Tokenize captions into WordPiece subwords
tokenized_captions = [tokenizer.encode(caption).tokens for caption in captions]

# Create vocab mappings
SUBWORD2IDX = {token: i for i, token in enumerate(tokenizer.get_vocab())}
IDX2SUBWORD = {i: token for token, i in SUBWORD2IDX.items()}
NUM_SUBWORDS = len(SUBWORD2IDX)


# Compute the maximum number of subwords in any caption
SUBWORD_MAX_LEN = max(len(tokens) for tokens in tokenized_captions) + 5 

# Print an example
#print("Example caption:", captions[0])
#print("Tokenized:", tokenized_captions[0])
#print("Word to Index:", [SUBWORD2IDX[word] for word in tokenized_captions[0]])

