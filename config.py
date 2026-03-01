import os
import torch


class Config:
    # --- Path Settings ---
    PROJECT_DIR = r'c:/Users/jfbaa/OneDrive/Documents/re-idversion2'

    DATA_DIR = os.path.join(PROJECT_DIR, 'data')

    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_384x384')
    TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_384x384')

    # --- Model Settings ---
    MODEL_NAME = 'convnext_base'
    IMG_SIZE = (384, 384)
    BATCH_SIZE = 32
    EMBEDDING_DIM = 512

    # --- Reranking Settings ---
    USE_RERANKING = True
    K1 = 20
    K2 = 6
    LAMBDA_VALUE = 0.3

    # --- Weights Path ---
    MODEL_WEIGHT_PATH = os.path.join(PROJECT_DIR, 'jaguar_resnet50_arcface_ep15.pth')

    # --- Device ---
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
