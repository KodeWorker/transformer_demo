import torch
from torch.nn import MSELoss
from torch.utils.data import random_split, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import WineReviewDataset
from model import SentimentRegressor
from tqdm import tqdm

def train_epoch(
      model,
      data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler):
      model = model.train()
      losses = []
      for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        loss = loss_fn(outputs.view(-1).float(), targets.view(-1).float())
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
      return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      loss = loss_fn(outputs.view(-1).float(), targets.view(-1).float())
      losses.append(loss.item())
  return np.mean(losses)

if __name__ == "__main__":
    RANDOM_SEED = 42
    PRE_TRAINED_MODEL_NAME = "bert-base-cased"
    VAL_RATIO = 0.2
    N_WORKERS = 4
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    
    review_path = "./data/winemag-data-130k-v2.csv"
    save_model_path = "./wine_review_model_state.bin"
    
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    # +++ setup wine review dataset +++
    review_df = pd.read_csv(review_path)    
    reviews = review_df["description"]
    targets = review_df["points"] / 100 # rating from 0 to 100 -> 0.0 to 1.0
    
    dataset = WineReviewDataset(reviews, targets, tokenizer)
    # --- setup wine review dataset ---
    
    valid_length = int(len(dataset)*VAL_RATIO)
    train_length = len(dataset) - valid_length
    
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
    
    print(f"#. training dataset: {len(train_dataset)}")
    print(f"#. validation dataset: {len(valid_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS)
    
    model = SentimentRegressor(PRE_TRAINED_MODEL_NAME)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=total_steps
    )
    
    loss_fn = MSELoss().to(device)
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        print(f"Train loss: {train_loss}")
    
        eval_loss = eval_model(model, valid_loader, loss_fn, device)
        print(f"Eval loss: {eval_loss}")
    
    torch.save(model.state_dict(), save_model_path)
    