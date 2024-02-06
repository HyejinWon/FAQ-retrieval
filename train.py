from sentence_transformers import SentenceTransformer, models, losses, LoggingHandler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import argparse
import random
import numpy as np
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader
import math
import utils
import os

def set_seed(args):
    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="klue/roberta-large")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--faq_pool_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--output_prefix", type=str, default="kor_nli_")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", type=str, default="./model_roberta_large")
    args = parser.parse_args()

    # Set seed
    set_seed(args)

    # Configure logger
    logging.basicConfig(
        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()]
    )

    # Define SentenceTransformer model
    word_embedding_model = models.Transformer(args.model_name_or_path, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Read FAQ pool data
    faq_pool = pd.read_csv(args.faq_pool_path)

    # Read train / dev data
    logging.info("Read train / dev data")
    train_data = utils.load_nli_sample_train(args.train_data_path, faq_pool=faq_pool) 

    logging.info(f"train data {len(train_data)} sample is made")
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    logging.info(f"{len(train_dataloader)} sample is made")
    
    # Configure the training.
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    dev_samples = utils.load_nli_samples(args.dev_sample_path, faq_pool=faq_pool)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.batch_size, name='sts-dev')
    
    train_loss = losses.ContrastiveLoss(model=model) # ContrastiveLoss / SoftmaxLoss / CosineSimilarityLoss

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=args.num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=args.model_save_path)
