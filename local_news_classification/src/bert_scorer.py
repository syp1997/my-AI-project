import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class BertScorer:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        logger.info('use device: {}'.format(self.device))
        
    def generate_score(self, bert_score_root, test=False):
        model = self.model
        model.eval()
    
        all_y_pred = []
        pbar = tqdm(enumerate(self.data_loader),total=len(self.data_loader))
        if test:
            for it, (text_ids, _, _) in pbar:
                # place data on the correct device
                text_ids = text_ids.to(self.device)
                with torch.set_grad_enabled(False):
                    y_pred = torch.sigmoid(model(text_ids))
                    all_y_pred.extend(y_pred)
                pbar.set_description(f"Bert Score Progress")
        else:
            for it, (text_ids, _, _, _) in pbar:
                # place data on the correct device
                text_ids = text_ids.to(self.device)
                with torch.set_grad_enabled(False):
                    y_pred = torch.sigmoid(model(text_ids))
                    all_y_pred.extend(y_pred)
                pbar.set_description(f"Bert Score Progress")
        all_y_pred = torch.stack(all_y_pred, dim=0)
        torch.save(all_y_pred, bert_score_root)
        return all_y_pred