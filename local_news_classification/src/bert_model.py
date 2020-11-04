import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BertConfig:
    """ base mdoel config """
    output_size = 1 # local(1) or non-local(0)
    dropout_prob = 0.1
    
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        for k, v in kwargs.items():
            setattr(self, k, v)
            

class Bert(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.ln = nn.LayerNorm(self.bert.pooler.dense.weight.shape[0], eps=1e-12)
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
            
        # log model all parameters and parameters need training
        logger.info('{} : all params: {:4f}M'.format(self._get_name(), sum(p.numel() for p in self.parameters()) / 1000 / 1000))
        logger.info('{} : need grad params: {:4f}M'.format(self._get_name(), sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000 / 1000))

    def forward(self, input_ids, labels=None, token_type_ids=None, attention_mask=None):
        
        _, bert_output = self.bert(input_ids, token_type_ids, attention_mask,)
        bert_output = self.ln(bert_output)
        x = self.dropout(bert_output)
        y_pred = self.fc(x).squeeze(-1)
        
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels)
            return y_pred, loss
        else:
            return y_pred 