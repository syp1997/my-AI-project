import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np


class ModelConfig:
    """ base mdoel config """
    output_size = 1 # local(1) or non-local(0)
    dropout_prob = 0.1
    
    def __init__(self, model_name, entity_vector, en_embd_dim, en_hidden_size1, en_hidden_size2, 
                 en_score_dim, **kwargs):
        self.model_name = model_name
        self.entity_vector = entity_vector
        self.en_embd_dim = en_embd_dim
        self.en_hidden_size1 = en_hidden_size1
        self.en_hidden_size2 = en_hidden_size2
        self.en_score_dim = en_score_dim
        for k, v in kwargs.items():
            setattr(self, k, v)
            

class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.ln = nn.LayerNorm(self.bert.pooler.dense.weight.shape[0], eps=1e-12)
        self.use_en_encoder = config.use_en_encoder
        if self.use_en_encoder: # if use entity infomation
            self.en_encoder = EntityEncoder(config)
            self.dropout = nn.Dropout(config.dropout_prob)
            self.fc = nn.Linear(self.bert.pooler.dense.weight.shape[0]+config.en_hidden_size2+config.en_score_dim, config.output_size)
        else:
            self.dropout = nn.Dropout(config.dropout_prob)
            self.fc = nn.Linear(self.bert.pooler.dense.weight.shape[0], config.output_size)
        
    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def fix_layer_grad(self, fix_layer=11):
        # do not train bert embedding layer
        for par in self.bert.embeddings.parameters(): 
            par.requires_grad = False
        # only train last(11th) bert encode layer
        for par in self.bert.encoder.layer[:fix_layer].parameters(): 
            par.requires_grad = False
            
        # print model all parameters and parameters need training
        print('{} : all params: {:4f}M'.format(self._get_name(), sum(p.numel() for p in self.parameters()) / 1000 / 1000))
        print('{} : need grad params: {:4f}M'.format(self._get_name(), sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000 / 1000))

    def forward(self, input_ids, entity_ids=None, entity_length=None, entity_score=None, 
                entity_embeds=None, labels=None, token_type_ids=None, attention_mask=None):
        _, bert_output = self.bert(input_ids, token_type_ids, attention_mask,)
        bert_output = self.ln(bert_output)
        if self.use_en_encoder: # if use entity infomation
            en_encoder_output = self.en_encoder(entity_ids, entity_length, entity_score, entity_embeds)
            x = torch.cat((bert_output,  en_encoder_output),dim=1)
        else:
            x = bert_output
        x = self.dropout(x)
        y_pred = self.fc(x).squeeze(-1)
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            return y_pred, loss
        else:
            return y_pred 
        
        
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

    def forward(self, entity_ids, entity_length, entity_score, entity_embeds):
        
        if entity_embeds is None:
            entity_embeds = self.en_embeddings(entity_ids)
        
        x = self.ln1(entity_embeds)
        x = self.dropout(x)
        
        x = self.mlp(x) # batch_size * entity_num * embd_dim
        x = self.single_pool(x, entity_length) #batch_size * embd_dim
        x = self.ln2(x)
        x = torch.cat((x, entity_score),dim=1)
        
        return x
    
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