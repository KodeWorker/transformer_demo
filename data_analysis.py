from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    PRE_TRAINED_MODEL_NAME = "bert-base-cased"
    N_BINS = 100
    review_path = "./data/winemag-data-130k-v2.csv"
    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    review_df = pd.read_csv(review_path)
    
    reviews = review_df["description"]
    targets = review_df["points"]
    
    # review token length    
    token_lens = []
    for txt in tqdm(reviews):
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))
    
    plt.figure()
    plt.title(os.path.basename(review_path))
    plt.hist(token_lens, bins=N_BINS)
    plt.show()
    