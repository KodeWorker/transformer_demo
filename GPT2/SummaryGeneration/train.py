import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataset import AmazonReviewV1, filter_nan
from dataset import AMAZON_REVIEW_LENGTH, AMAZON_SUMMARY_LENGTH, BOS, EOS, PAD
from torch.utils.data import random_split, DataLoader

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel,  GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

def train_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scheduler):
    
    model = model.train()
    losses = []
    
    for d in tqdm(data_loader):
    
        input_ids = d["review"]["input_ids"].to(device)
        attention_mask = d["review"]["attention_mask"].to(device)
        labels = d["summary"]["input_ids"].to(device)
        
        model.zero_grad()
        
        outputs = model(
          input_ids=input_ids,
          labels=labels,
          attention_mask=attention_mask,
          token_type_ids=None
        )
        
        loss = outputs[0]
        losses.append(loss.item())
        
        loss.backward()        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return np.mean(losses)
        
if __name__ == "__main__":

    RANDOM_SEED = 42    
    VAL_RATIO = 0.2
    N_WORKERS = 4
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP = 1e2
    EPSILON = 1e-8
    
    PRETRAINED_MODEL_NAME = "gpt2"
    REVIEW_PATH = "../../data/amazon_fine_food_review/Reviews.csv"
    
    # initialize random state
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build dataset
    df = pd.read_csv(REVIEW_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.pad_token = tokenizer.eos_token
     
    len_dict = {"review": AMAZON_REVIEW_LENGTH,
                        "summary": AMAZON_SUMMARY_LENGTH}
                        
    reviews, summaries = filter_nan(df)
    dataset = AmazonReviewV1(reviews, summaries, tokenizer, len_dict, BOS, EOS)
    
    valid_length = int(len(dataset)*VAL_RATIO)
    train_length = len(dataset) - valid_length
    
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
    
    print(f"#. training dataset: {len(train_dataset)}")
    print(f"#. validation dataset: {len(valid_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    
    # build model
    print(f"model embedding size: {len(tokenizer)}")
    
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
    model.resize_token_embeddings(len(tokenizer))
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP, num_training_steps = total_steps)
    
    model = model.to(device)
    
    # training phase
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        print(f"Train loss: {train_loss}")
        