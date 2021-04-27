if __name__ == "__main__":
    
    import pandas as pd
    import numpy as np
    from transformers import BertTokenizer
    import matplotlib.pyplot as plt
    
    PRETRAINED_MODEL_NAME = "bert-base-cased"
    REVIEW_PATH = "../../data/amazon_fine_food_review/Reviews.csv"
    
    df = pd.read_csv(REVIEW_PATH)
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    
    summaries = df["Summary"].values.tolist()
    reviews = df["Text"].values.tolist()
    
    n_summary_tokens = []
    summary_nan_count = 0
    for summary in summaries:
        try:
            n_summary_tokens.append(len(summary))
        except:
            summary_nan_count += 1
    print("-"*30)
    print(f"summary_nan_count: {summary_nan_count}")
    print(f"mean: {np.mean(n_summary_tokens)}, std: {np.std(n_summary_tokens)}")
    print(f"min:{np.min(n_summary_tokens)}, max: {np.max(n_summary_tokens)}")
    print(f"mean + 2 * sigma = {np.ceil(np.mean(n_summary_tokens) + 2*np.std(n_summary_tokens))}")
    
    n_review_tokens = []
    review_nan_count = 0
    for review in reviews:
        try:
            n_review_tokens.append(len(review))
        except:
            review_nan_count += 1
    print("-"*30)
    print(f"review_nan_count: {review_nan_count}")
    print(f"mean: {np.mean(n_review_tokens)}, std: {np.std(n_review_tokens)}")
    print(f"min:{np.min(n_review_tokens)}, max: {np.max(n_review_tokens)}")
    print(f"mean + 2 * sigma = {np.ceil(np.mean(n_review_tokens) + 2*np.std(n_review_tokens))}")
    
    plt.figure()
    plt.hist(n_review_tokens, bins=100, label="review")
    plt.hist(n_summary_tokens, bins=100, label="summary")    
    plt.legend()
    plt.show()