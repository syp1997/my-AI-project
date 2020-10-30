import torch
import numpy as np
from tqdm import tqdm
import math
import logging

logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # may useful optimize method
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False # optimize method
    warmup_tokens = 375e6 # use this to train model from a lower learning rate
    final_tokens = 260e9 # all tokens during whole training process
    # checkpoint settings
    ckpt_path = 'local-likely-model.pt' # save model path
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            print(k,v)
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        print('use device:', self.device)
        
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
        
    def binary_accuracy(self, preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_loader if is_train else self.test_loader
            
            losses = []
            all_y = []
            all_y_pred = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (text_ids, entity_ids, entity_length, entity_score, y) in pbar:
                # place data on the correct device
                text_ids = text_ids.to(self.device)
                entity_ids = entity_ids.to(self.device)
                entity_length = entity_length.to(self.device)
                entity_score = entity_score.to(self.device)
                y = y.to(self.device)
                # forward the model
                with torch.set_grad_enabled(is_train):
                    y_pred, loss = model(text_ids, entity_ids, entity_length, entity_score, None, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    step_score = self.binary_accuracy(y_pred, y)
                    all_y.extend(y)
                    all_y_pred.extend(y_pred)
                
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. score {step_score:.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                all_y = torch.stack(all_y, dim=0)
                all_y_pred = torch.stack(all_y_pred, dim=0)
                test_score = self.binary_accuracy(all_y_pred, all_y)
                logger.info("test loss: %f", test_loss)
                logger.info("test score: %f", test_score)
                return test_loss

        self.tokens = 0 # counter used for learning rate decay
        best_loss = float('inf')
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_loader is not None:
                test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_loader is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()