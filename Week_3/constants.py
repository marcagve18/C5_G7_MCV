from pathlib import Path

ROOT_PATH = Path("/ghome/c5mcv07/C5_G7_MCV/Week_3")

DATASET_PATH = ROOT_PATH / "dataset"
IMAGES_PATH = DATASET_PATH / "Food Images"
ANNOTATIONS_PATH = DATASET_PATH / "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"

OUTPUTS_PATH = ROOT_PATH / "outputs"

CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(CHARS)
IDX2CHAR = {k: v for k, v in enumerate(CHARS)}
CHAR2IDX = {v: k for k, v in enumerate(CHARS)}
TEXT_MAX_LEN = 201
