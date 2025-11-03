import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_PARENT_DIR = PROJECT_ROOT.parent / "Sentiment_Analysis_Data"

RAW_DATA_DIR = DATA_PARENT_DIR / "raw"
CLEAN_DATA_DIR = DATA_PARENT_DIR / "clean"

MODELS_DIR = PROJECT_ROOT / "models"


RAW_IMDB_TRAIN_PATH = RAW_DATA_DIR / "raw_imdb_train.csv"
RAW_IMDB_TEST_PATH = RAW_DATA_DIR / "raw_imdb_test.csv"
RAW_IMDB_UNSUPERVISED_PATH = RAW_DATA_DIR / "raw_imdb_unsupervised.csv"

RAW_RT_TRAIN_PATH = RAW_DATA_DIR / "raw_rt_train.csv"
RAW_RT_VAL_PATH = RAW_DATA_DIR / "raw_rt_val.csv"
RAW_RT_TEST_PATH = RAW_DATA_DIR / "raw_rt_test.csv"

CLEAN_IMDB_TRAIN_PATH = CLEAN_DATA_DIR / "clean_imdb_train.csv"
CLEAN_IMDB_VAL_PATH = CLEAN_DATA_DIR / "clean_imdb_val.csv"
CLEAN_IMDB_TEST_PATH = CLEAN_DATA_DIR / "clean_imdb_test.csv"
CLEAN_IMDB_UNSUPERVISED_PATH = CLEAN_DATA_DIR / "clean_imdb_unsupervised.csv"


CLEAN_RT_TRAIN_PATH = CLEAN_DATA_DIR / "clean_rt_train.csv"
CLEAN_RT_VAL_PATH = CLEAN_DATA_DIR / "clean_rt_val.csv"
CLEAN_RT_TEST_PATH = CLEAN_DATA_DIR / "clean_rt_test.csv"

TEXT_COL = 'text'
LABEL_COL = 'label'

W2V_MODEL_PATH = MODELS_DIR / "word2vec_full.model"
W2V_VECTORS_PATH = MODELS_DIR / "word2vec_vectors.kv"
BEST_LSTM_IMDB_PATH = MODELS_DIR / "best_lstm_imdb.pt"
BEST_ROBERTA_IMDB_PATH = MODELS_DIR / "best_roberta_imdb.pt"
BEST_LSTM_RT_PATH = MODELS_DIR / "best_lstm_rt.pt"
BEST_ROBERTA_RT_PATH = MODELS_DIR / "best_roberta_rt.pt"

RANDOM_SEED = 42