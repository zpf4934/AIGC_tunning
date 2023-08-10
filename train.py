# -*- coding:utf-8 -*-
"""
# File       : train.py
# Time       ：2023/6/27 17:52
# Author     ：andy
# version    ：python 3.9
"""
import json
import time
import warnings
from datetime import timedelta

import deepspeed
import jieba
import numpy as np
import torch
from accelerate.utils import set_seed
from deepspeed.accelerator import get_accelerator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from transformers import HfArgumentParser

from arguments import ModelArguments, TrainingArguments
from tunning import TUNING

warnings.filterwarnings('ignore')


def train(train_dataloader, eval_dataloader, tuning):
    if tuning.config.distributed != 'deepspeed':
        tuning.load_model()
        tuning.create_optimizer(train_dataloader)
    tuning.model = tuning.model.to(tuning.device)
    if tuning.config.distributed == 'accelerate':
        if tuning.config.seed is not None:
            set_seed(tuning.config.seed)
        tuning.accelerator.print(tuning.model.print_trainable_parameters())
        tuning.model, tuning.optimizer, train_dataloader, eval_dataloader, tuning.lr_scheduler = tuning.accelerator.prepare(
            tuning.model, tuning.optimizer, train_dataloader, eval_dataloader, tuning.lr_scheduler
        )
    else:
        tuning.model.print_trainable_parameters()
        if tuning.config.seed is not None:
            torch.manual_seed(tuning.config.seed)
            torch.cuda.manual_seed_all(tuning.config.seed)
            np.random.seed(tuning.config.seed)
            torch.backends.cudnn.deterministic = True
    start_time = time.time()
    best_loss = float("inf")
    total_batch = 0
    last_improve_batch = 0
    for epoch in range(tuning.config.epochs):
        tuning.model.train()
        total_loss = 0
        all_predictions = None
        all_references = None
        get_accelerator().empty_cache()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(tuning.device) for k, v in batch.items()}
            outputs = tuning.model(**batch)
            loss = outputs.loss
            if not loss.requires_grad:
                loss.requires_grad = True
            total_loss += loss.detach().float()
            if tuning.config.distributed == 'deepspeed':
                tuning.model.backward(loss)
                tuning.model.step()
            else:
                if tuning.config.distributed == 'accelerate':
                    tuning.accelerator.backward(loss)
                else:
                    loss.backward()
                    # tuning.model.backward(loss)
                tuning.optimizer.step()
                tuning.lr_scheduler.step()
                tuning.optimizer.zero_grad()
            predictions = torch.argmax(outputs.logits, -1)
            references = batch['labels']
            all_predictions = predictions if all_predictions is None else torch.cat([all_predictions, predictions])
            all_references = references if all_references is None else torch.cat([all_references, references])

        train_epoch_loss = total_loss / len(train_dataloader)
        if tuning.config.distributed == 'accelerate':
            all_predictions, all_references = tuning.accelerator.gather_for_metrics((all_predictions, all_references))
        train_metrics = compute_metrics(all_predictions.detach().cpu().numpy(), all_references.detach().cpu().numpy(),
                                        tuning)

        eval_loss, eval_metrics = evaluate(eval_dataloader, tuning)
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        if eval_epoch_loss < best_loss:
            best_loss = eval_epoch_loss
            improve = '*'
            save_model(tuning)
            last_improve_batch = total_batch
        else:
            improve = ''
        total_batch += 1
        if torch.is_tensor(train_epoch_loss):
            train_epoch_loss = train_epoch_loss.item()
        if torch.is_tensor(eval_epoch_loss):
            eval_epoch_loss = eval_epoch_loss.item()
        metric_str = "Epoch:{0}/{1}, Train Rouge:{2:.4f}, Train Bleu:{3:.4f}, Train Loss:{4:.4f}, Eval Rouge:{5:.4f}, " \
                     "Eval Bleu:{6:.4f}, Eval Loss:{7:.4f}, Time: {8} {9}"
        metric_str = metric_str.format(epoch + 1, tuning.config.epochs, train_metrics['rouge-1'], train_metrics['bleu'],
                                       train_epoch_loss, eval_metrics['rouge-1'], eval_metrics['bleu'], eval_epoch_loss,
                                       timedelta(seconds=int(round(time.time() - start_time))), improve)
        if tuning.config.distributed == 'accelerate':
            tuning.accelerator.print(metric_str)
        else:
            print(metric_str)
        print("Epoch:{0}/{1}, Train Rouge:{2:.4f}, Train Bleu:{3:.4f}, Train Loss:{4:.4f}, Eval Rouge:{5:.4f}, "
              "Eval Bleu:{6:.4f}, Eval Loss:{7:.4f}, Time: {8} {9}"
              "".format(epoch + 1, tuning.config.epochs, train_metrics['rouge-1'], train_metrics['bleu'],
                        train_epoch_loss, eval_metrics['rouge-1'], eval_metrics['bleu'],
                        eval_epoch_loss, timedelta(seconds=int(round(time.time() - start_time))), improve))
        if total_batch - last_improve_batch > tuning.config.early_stop_step:
            print("No optimization for a long time, Stop......")
            return


def evaluate(eval_dataloader, tuning):
    tuning.model.eval()
    eval_loss = 0
    eval_metrics = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu": []
    }
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(tuning.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = tuning.model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        predictions = torch.argmax(outputs.logits, -1)
        references = batch['labels']
        if tuning.config.distributed == 'accelerate':
            predictions, references = tuning.accelerator.gather_for_metrics((predictions, references))
        batch_metrics = compute_metrics(predictions.detach().cpu().numpy(),
                                        references.detach().cpu().numpy(), tuning)
        for key, value in eval_metrics.items():
            value.append(batch_metrics[key])
    for k, v in eval_metrics.items():
        eval_metrics[k] = round(float(np.mean(v)), 4)
    return eval_loss, eval_metrics


def compute_metrics(preds, labels, tuning):
    decoded_preds = tuning.tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tuning.tokenizer.pad_token_id)
    decoded_labels = tuning.tokenizer.batch_decode(labels, skip_special_tokens=True)
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        while ' ' in hypothesis:
            hypothesis.remove(' ')
        while ' ' in reference:
            reference.remove(' ')
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        try:
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        except:
            continue
        result = scores[0]
        for k, v in result.items():
            score_dict[k].append(round(v["f"], 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu"].append(round(bleu_score, 4))
    for k, v in score_dict.items():
        score_dict[k] = round(float(np.mean(v)), 4)
    return score_dict


def save_model(tuning):
    save_path = "outputs/" + training_args.save_path
    if tuning.config.distributed == 'accelerate':
        tuning.accelerator.wait_for_everyone()
        unwrapped_model = tuning.accelerator.unwrap_model(tuning.model)
        unwrapped_model.save_pretrained(save_path, save_function=tuning.accelerator.save,
                                        state_dict=tuning.accelerator.get_state_dict(tuning.model))
    else:
        tuning.model.save_pretrained(save_path)


def init_deepspeed(tuning):
    deepspeed.init_distributed()
    tuning.load_model()
    model_parameters = filter(lambda p: p.requires_grad, tuning.model.parameters())
    tuning.model, _, _, _ = deepspeed.initialize(args=model_args, model=tuning.model, model_parameters=model_parameters,
                                                 optimizer=None)


def main():
    with open(training_args.dataset_name, 'r') as fr:
        data = json.load(fr)
    tuning = TUNING[training_args.tuning_method](model_args)
    dataset = tuning.tokenize(data[:20000])
    train_loader = tuning.dataloader(dataset[:int(len(dataset) * 0.7)])
    eval_loader = tuning.dataloader(dataset[int(len(dataset) * 0.7):])
    if tuning.config.distributed == 'deepspeed':
        init_deepspeed(tuning)
    train(train_loader, eval_loader, tuning)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    main()
