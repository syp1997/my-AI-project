import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EntityModelConfig:
    """ base mdoel config """
    output_size = 1 # local(1) or non-local(0)
    dropout_prob = 0.1
    
    def __init__(self, entity_vector, en_embd_dim, en_hidden_size1, en_hidden_size2, **kwargs):
        self.entity_vector = entity_vector
        self.en_embd_dim = en_embd_dim
        self.en_hidden_size1 = en_hidden_size1
        self.en_hidden_size2 = en_hidden_size2
        for k, v in kwargs.items():
            setattr(self, k, v)
            

class EntityEncoder(nn.Module):
    """ Encode entities to generate single presentation """

    def __init__(self, config):
        super().__init__()
        self.en_embeddings = nn.Embedding.from_pretrained(config.entity_vector,freeze=True)

        self.ln1 = nn.LayerNorm(config.en_embd_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.mlp = nn.Sequential(
            nn.Linear(config.en_embd_dim, config.en_hidden_size1),
            nn.GELU(),
            nn.Linear(config.en_hidden_size1, config.en_hidden_size2),
            nn.Dropout(config.dropout_prob), # maybe useful
        )
        self.ln2 = nn.LayerNorm(config.en_hidden_size2, eps=1e-12)
        self.fc = nn.Linear(config.en_hidden_size2, config.output_size)
        
    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, entity_ids, entity_length, entity_embeds=None, labels=None):
        
        if entity_embeds is None:
            entity_embeds = self.en_embeddings(entity_ids)
        
        x = self.ln1(entity_embeds)
        x = self.dropout(x)
        
        x = self.mlp(x) # batch_size * entity_num * embd_dim
        x = self.single_pool(x, entity_length) #batch_size * embd_dim
        x = self.ln2(x)
        y_pred = self.fc(x).squeeze(-1)
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            return y_pred, loss
        else:
            return y_pred 
    
    
    # do this because of different entity length
    def single_pool(self, x, x_length):
        all_pool_out = []
        for i in range(x.shape[0]):
            if x_length[i] == 0:
                 x_length[i] += 1
            single_data = x[i][:x_length[i]].unsqueeze(0)
            pool_out = F.max_pool2d(single_data, (single_data.shape[1], 1)).squeeze(1)
            all_pool_out.append(pool_out)
        x = torch.cat(all_pool_out,dim=0)
        return x