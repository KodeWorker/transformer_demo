from torch.nn import Module, Dropout, Sequential, Linear, LeakyReLU
from transformers import AlbertModel

class SentimentRegressor(Module):
    
    def __init__(self, pretrain_model_name):
        super( SentimentRegressor, self).__init__()
        
        self.bert = AlbertModel.from_pretrained(pretrain_model_name)
        
        self.drop = Dropout(p=0.3)
        
        self.net = Sequential(
            Linear(self.bert.config.hidden_size, 128),
            LeakyReLU(),
            Linear(128, 32),
            LeakyReLU(),
            Linear(32, 1),
        )
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(output.pooler_output)
        return self.net(output)