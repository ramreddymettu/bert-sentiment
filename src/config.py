import transformers

DEVICE = "cpu"

FEATURE_EXTRACTING = True

# Max lenght of sequence
MAX_LENGTH = 512

#Train batch size
TRAIN_BATCH_SIZE = 8

#Train data
TRAIN_FILE = "../input/IMDB Dataset.csv"

# Test batch size
TEST_BATCH_SIZE = 4

# Number of loops
EPOCHS = 1

#Model Path to save
MODEL_PATH = "../input/model.bin"

# Pre-trained model path
BERT_UNCASED_PATH = "../input/bert-base-uncased"

#Bert Tokenizer
TOKENIZERS = transformers.BertTokenizer.from_pretrained(
    BERT_UNCASED_PATH,
    do_lower = True
)