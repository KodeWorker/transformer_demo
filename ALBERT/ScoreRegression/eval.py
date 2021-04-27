import torch
import numpy as np
from model import SentimentRegressor
from transformers import AlbertTokenizer

if __name__ == "__main__":
    
    RANDOM_SEED = 42
    MAX_LEN = 150
    PRE_TRAINED_MODEL_NAME = "albert-base-v2"
    save_model_path = "../data/wine_review_albert.bin"
    
    # eval message
    review = "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity."
    
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SentimentRegressor(PRE_TRAINED_MODEL_NAME)
    model.load_state_dict(torch.load(save_model_path))
    model.to(device)
    model.eval()
    
    tokenizer = AlbertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    encoding = tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      truncation=True,
      max_length=MAX_LEN,
      return_token_type_ids=False,
      #pad_to_max_length=True,
      padding="max_length",
      return_attention_mask=True,
      return_tensors='pt',
    )
    
    inputs = {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
    }
    
    input_ids = torch.unsqueeze(inputs["input_ids"], 0).to(device)
    attention_mask = torch.unsqueeze(inputs["attention_mask"], 0).to(device)
    score = model(input_ids=input_ids, 
                               attention_mask=attention_mask).cpu().detach().numpy()[0, 0]
    
    print(f"review: {review}")
    print(f"score: {score}")
    