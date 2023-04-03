# !pip install -qqq datasets transformers --upgrade

import urllib.request
import time
import json

import numpy as np
import os
import matplotlib.pyplot as plt

from typing import List, Dict, Union, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModel
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer

from encoder_decoder_solution import EncoderDecoder
from transformer_solution import Transformer
from transformers import AutoTokenizer

tokenizer_name = "bert-base-uncased" #@param {type: "string"}
sample_max_length = 256 #@param {type:"slider", min:32, max:512, step:1}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.random.manual_seed(0)
import wandb
import random

class Collate:
    def __init__(self, tokenizer: str, max_len: int) -> None:
        self.tokenizer_name = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Union[str, int]]]) -> Dict[str, torch.Tensor]:
        texts = list(map(lambda batch_instance: batch_instance["title"], batch))
        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        
        labels = list(map(lambda batch_instance: int(batch_instance["label"]), batch))
        labels = torch.LongTensor(labels)
        return dict(tokenized_inputs, **{"labels": labels})

class ReviewClassifier(nn.Module):
    def __init__(self, backbone: str, backbone_hidden_size: int, nb_classes: int):
        super(ReviewClassifier, self).__init__()
        self.backbone = backbone
        self.backbone_hidden_size = backbone_hidden_size
        self.nb_classes = nb_classes
        self.back_bone = AutoModel.from_pretrained(
            self.backbone,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.classifier = torch.nn.Linear(self.backbone_hidden_size, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        back_bone_output = self.back_bone(input_ids, attention_mask=attention_mask)
        hidden_states = back_bone_output[0]
        pooled_output = hidden_states[:, 0]  # getting the [CLS] token
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits

class ReviewClassifierLSTM(nn.Module):
    def __init__(self, nb_classes: int, encoder_only: bool = False, dropout=0.5):
        super(ReviewClassifierLSTM, self).__init__()
        self.nb_classes = nb_classes
        self.encoder_only = encoder_only
        self.back_bone = EncoderDecoder(dropout=dropout, encoder_only=encoder_only) 
        self.classifier = torch.nn.Linear(256, self.nb_classes)
       
    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden_states, _ = self.back_bone(input_ids, attention_mask)
        pooled_output = hidden_states 
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits


class ReviewClassifierTransformer(nn.Module):
    def __init__(self, nb_classes: int, num_heads: int = 4, num_layers: int = 4, block: str="prenorm", dropout: float = 0.3):
        super(ReviewClassifierTransformer, self).__init__()
        self.nb_classes = nb_classes
        self.back_bone = Transformer(num_heads=num_heads, num_layers=num_layers, block=block, dropout=dropout)
        self.classifier = torch.nn.Linear(256, self.nb_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        back_bone_output = self.back_bone(input_ids, attention_mask)
        hidden_states = back_bone_output
        pooled_output = hidden_states
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, self.nb_classes),
                labels.view(-1),
            )
            return loss, logits
        return logits
    

def show_samples(n_samples_to_see, dataset_test):
    n_samples_to_see = 3 
    for i in range(n_samples_to_see):
        print("-"*30)
        print("title:", dataset_test[i]["title"])
        print("content:", dataset_test[i]["content"])
        print("label:", dataset_test[i]["label"])

def train_one_epoch(
    model: torch.nn.Module, training_data_loader: DataLoader, optimizer: torch.optim.Optimizer, logging_frequency: int, testing_data_loader: DataLoader, logger: dict):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0
    logging_loss = 0
    start_time = time.time()
    mini_start_time = time.time()
    for step, batch in enumerate(training_data_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        logging_loss += loss.item()

        if (step + 1) % logging_frequency == 0:
            freq_time = time.time()-mini_start_time
            logger['train_time'].append(freq_time+logger['train_time'][-1])
            logger['train_losses'].append(logging_loss/logging_frequency)
            print(f"Training loss @ step {step+1}: {logging_loss/logging_frequency}")
            eval_acc, eval_loss, eval_time = evaluate(model, testing_data_loader)
            logger['eval_accs'].append(eval_acc)
            logger['eval_losses'].append(eval_loss)
            logger['eval_time'].append(eval_time+logger['eval_time'][-1])
            wandb.log({metric:logger[metric][-1] for metric in logger})
            logging_loss = 0
            mini_start_time = time.time()

    return epoch_loss / len(training_data_loader), time.time()-start_time


def evaluate(model: torch.nn.Module, test_data_loader: DataLoader):
    model.eval()
    model.to(device)
    eval_loss = 0
    correct_predictions = {i: 0 for i in range(2)}
    total_predictions = {i: 0 for i in range(2)}
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(test_data_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            eval_loss += loss.item()
            predictions = np.argmax(outputs[1].detach().cpu().numpy(), axis=1)
            for target, prediction in zip(batch["labels"].cpu().numpy(), predictions):
                if target == prediction:
                    correct_predictions[target] += 1
                total_predictions[target] += 1
    accuracy = (100.0 * sum(correct_predictions.values())) / sum(total_predictions.values())
    model.train()
    return accuracy,  eval_loss / len(test_data_loader), time.time() - start_time

def get_logger(model):
  logger = dict()
  logger['train_time'] = [0]
  logger['eval_time'] = [0]
  logger['train_losses'] = []
  logger['eval_accs'] = []
  logger['eval_losses'] = []
  logger['parameters'] = sum([p.numel() for p in model.back_bone.parameters() if p.requires_grad])
  return logger

def put_in_dictionary(logger, train_loss, train_time, eval_loss, eval_time, eval_acc):
  logger["total_train_loss"] = train_loss
  logger["total_train_time"] = train_time
  logger["final_eval_loss"] = eval_loss
  logger["final_eval_time"] = eval_time
  logger["final_eval_acc"] = eval_acc
  logger['train_time'] = logger['train_time'][1:]
  logger['eval_time'] = logger['eval_time'][1:]
  return logger

def save_logs(dictionary, log_dir, exp_id):
  log_dir = os.path.join(log_dir, exp_id)
  os.makedirs(log_dir, exist_ok=True)
  # Log arguments
  with open(os.path.join(log_dir, "args.json"), "w") as f:
    json.dump(dictionary, f, indent=2)

def main():
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_train = load_dataset("amazon_polarity", split="train", cache_dir="assignment/data")
    dataset_test = load_dataset("amazon_polarity", split="test[:1000]", cache_dir="assignment/data")
    collate = Collate(tokenizer=tokenizer_name, max_len=sample_max_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> Device selected: {device}")

    nb_epoch = 1
    batch_size = 512
    logging_frequency = 5 
    learning_rate = 1e-5

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate)
    for i in range(1, 7):
        experimental_setting = i
        # 4 experimental settings
        torch.random.manual_seed(0)
        print("Initiating setting",experimental_setting)

        #Choosing experimental setting and model
        if experimental_setting == 1:
            exp_args = dict(nb_classes=2, dropout=0.3, encoder_only=True)            
            model = ReviewClassifierLSTM(**exp_args)
            exp_args['model_class'] = "ReviewClassifierLSTM"
        elif experimental_setting == 2:
            exp_args = dict(nb_classes=2, dropout=0.3, encoder_only=False)
            model = ReviewClassifierLSTM(**exp_args)
            exp_args['model_class'] = "ReviewClassifierLSTM"
        elif experimental_setting == 3:
            exp_args = dict(nb_classes=2, num_heads=4, num_layers=2, block='prenorm', dropout=0.3)
            model = ReviewClassifierTransformer(**exp_args)
            exp_args['model_class'] = "ReviewClassifierTransformer"
        elif experimental_setting == 4:
            exp_args = dict(nb_classes=2, num_heads=4, num_layers=4, block='prenorm', dropout=0.3)
            model = ReviewClassifierTransformer(**exp_args)
            exp_args['model_class'] = "ReviewClassifierTransformer"
        elif experimental_setting == 5:
            exp_args = dict(nb_classes=2, num_heads=4, num_layers=2, block='postnorm', dropout=0.3)
            model = ReviewClassifierTransformer(**exp_args)
            exp_args['model_class'] = "ReviewClassifierTransformer"            
        elif experimental_setting == 6: 
            exp_args = dict(backbone="bert-base-uncased", backbone_hidden_size=768, nb_classes=2)
            model = ReviewClassifier(**exp_args)
            for parameter in model.back_bone.parameters():
                parameter.requires_grad= False
                logging_frequency = 703
            exp_args['model_class'] = "ReviewClassifierTransformer"

        #Initialising wandb experiment
        exp_args['experimental_setting'] = experimental_setting
        wandb.init(project="RepLearning - A2",config=exp_args)

        #Setting up the optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-8)
        model.to(device)
        logger = get_logger(model)
        
        #Training model for nb_epoch epochs
        for i in range(nb_epoch):
            train_loss, train_time = train_one_epoch(model, train_loader, optimizer, logging_frequency, test_loader, logger)
        
        #Evalutating model
        eval_acc, eval_loss, eval_time = evaluate(model, test_loader)
        print("Eval accuracy:",eval_acc,"Eval loss",eval_loss,"Eval time",eval_time)
        
        #Print logger values and save logs
        logger = put_in_dictionary(logger, train_loss, train_time, eval_loss, eval_time, eval_acc)
        print(f"    Epoch: {1} Loss/Test: {eval_loss}, Loss/Train: {train_loss}, Acc/Test: {eval_acc}, Train Time: {train_time}, Eval Time: {eval_time}")
        save_logs(logger, "assignment/log", str(experimental_setting))


if __name__ == "__main__":
    main()