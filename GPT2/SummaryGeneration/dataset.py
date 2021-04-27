from torch.utils.data import Dataset
from tqdm import tqdm

AMAZON_REVIEW_LENGTH = 1327
AMAZON_SUMMARY_LENGTH = 52
BOS = "<|startoftext|>"
EOS = "<|endoftext|>"
PAD = "<|pad|>"

class AmazonReviewV2(Dataset):

    def __init__(self, reviews, summaries, tokenizer, len_dict, bos_token, eos_token):
        super().__init__()
        self.reviews = reviews
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.len_dict = len_dict
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        self.encoding = []
        for index in tqdm(range(len(self.reviews))):
            review, summary = self.reviews[index], self.summaries[index]
            self.encoding.append(self.get_encoding(review, summary))
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, index):
        return  self.encoding[index]
    
    def get_encoding(self, review, summary):
        # review review
        review_encoding = self.tokenizer.encode_plus(
          self.bos_token + review + self.eos_token,
          add_special_tokens=True,
          truncation=True,
          max_length=self.len_dict["review"],
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        )
        # summary encoding
        summary_encoding = self.tokenizer.encode_plus(
          self.bos_token + summary + self.eos_token,
          add_special_tokens=True,
          truncation=True,
          max_length=self.len_dict["summary"],
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {"review": review_encoding,
                      "summary": summary_encoding}

class AmazonReviewV1(Dataset):

    def __init__(self, reviews, summaries, tokenizer, len_dict, bos_token, eos_token):
        super().__init__()
        self.reviews = reviews
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.len_dict = len_dict
        self.bos_token = bos_token
        self.eos_token = eos_token
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, index):
        review, summary = self.reviews[index], self.summaries[index]
        # review review
        review_encoding = self.tokenizer.encode_plus(
          self.bos_token + review + self.eos_token,
          add_special_tokens=True,
          truncation=True,
          max_length=self.len_dict["review"],
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        )
        # summary encoding
        summary_encoding = self.tokenizer.encode_plus(
          self.bos_token + summary + self.eos_token,
          add_special_tokens=True,
          truncation=True,
          max_length=self.len_dict["summary"],
          return_token_type_ids=False,
          padding="max_length",
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {"review": review_encoding,
                      "summary": summary_encoding}

def filter_nan(df):
    sel = df["Summary"].notnull()
    summaries = df["Summary"][sel].values.tolist()
    reviews = df["Text"][sel].values.tolist()
    return reviews, summaries

if __name__ == "__main__":
    
    import pandas as pd
    from transformers import GPT2Tokenizer
    
    len_dict = {"review": AMAZON_REVIEW_LENGTH,
                        "summary": AMAZON_SUMMARY_LENGTH}
    
    PRETRAINED_MODEL_NAME = "gpt2"
    REVIEW_PATH = "../../data/amazon_fine_food_review/Reviews.csv"
    
    df = pd.read_csv(REVIEW_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, bos_token=BOS, eos_token=EOS, pad_token=PAD)
    
    # build dataset
    reviews, summaries = filter_nan(df)
    dataset = AmazonReviewV1(reviews, summaries, tokenizer, len_dict, BOS, EOS)
    