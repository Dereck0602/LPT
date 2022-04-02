"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import argparse
import glob
import os
import pickle
import random
import re
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import XLMRobertaConfig,XLMRobertaForSequenceClassification, XLMRobertaTokenizer,XLMRobertaForMaskedLM
import math
import argparse
from utils.checkpoint import *
from utils.dataset import *
from torch.nn import CrossEntropyLoss

class ArgsParser(object):
     
     def __init__(self):

          parser = argparse.ArgumentParser()
          parser.add_argument("--src_language", default='en', type=str, required=True,
                              help="source language selected in the list: ['en','fr','jp','de']")
          parser.add_argument("--src_domain", default='en', type=str, required=True,
                              help="source language selected in the list: ['dvd', 'books', 'music','all']")
          parser.add_argument("--labelword_ensemble", default=None, type=str, required=False,
                              help="whether use labelword_ensemble")
          parser.add_argument("--label0", default=False, type=str, required=False,
                              help="the label word for label0")
          parser.add_argument("--label1", default=False, type=str, required=False,
                              help="the label word for label1")
          parser.add_argument("--train_type", default="finetune", type=str, required=True,
                              help="the type of training selected in the list: ['finetune','headtune','prompttune']")
          parser.add_argument("--data_type", default="all", type=str, required=True,
                              help="the number of data selected in the list: ['all','few']")
          parser.add_argument("--n_shot", default=32, type=int, required=True,
                              help="the number of few shot")
          parser.add_argument("--n_class", default=2, type=int, required=True,
                              help="the number of classes")                                        
          parser.add_argument("--head_type", default="classifier", type=str, required=True,
                              help="the type of training selected in the list: ['classifier','mlm']")
          parser.add_argument("--model_type", default=None, type=str, required=True,
                              help="Model type selected in the list: [xlmr] ")
          parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                              help="Path to pre-trained model or shortcut name selected in the list:")
          #parser.add_argument("--output_dir", default=None, type=str, required=True,
          #                    help="The output directory where the model checkpoints and predictions will be written.")
          parser.add_argument("--config_name", default=None, type=str,
                              help="Pretrained config name or path if not the same as model_name")
          parser.add_argument("--tokenizer_name", default=None, type=str,
                              help="Pretrained tokenizer name or path if not the same as model_name")
          parser.add_argument("--cache_dir", default=None, type=str,
                              help="Where do you want to store the pre-trained models downloaded from s3")
          parser.add_argument("--max_len", default=512, type=int,
                              help="The maximum total encoder sequence length."
                                   "Longer than this will be truncated, and sequences shorter than this will be padded.")
          parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                              help="Batch size per GPU/CPU for training.")
          parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                              help="Batch size per GPU/CPU for training.")
          parser.add_argument("--save_total_limit", default=-1, type=int,
                              help="maximum of checkpoint to be saved")
          parser.add_argument("--learning_rate", default=5e-5, type=float,
                              help="The initial learning rate for Adam.")
          parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                              help="Number of updates steps to accumulate before performing a backward/update pass.")
          parser.add_argument("--weight_decay", default=0.00, type=float,
                              help="Weight decay if we apply some.")
          parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                              help="Epsilon for Adam optimizer.")
          parser.add_argument("--max_grad_norm", default=1.0, type=float,
                              help="Max gradient norm.")
          parser.add_argument("--num_train_steps", default=-1, type=int,
                              help="set total number of training steps to perform")
          parser.add_argument("--num_train_epochs", default=10, type=int,
                              help="set total number of training epochs to perform (--num_training_steps has higher priority)")
          parser.add_argument("--num_warmup_steps", default=0, type=int,
                              help="Linear warmup over warmup_steps.")
          parser.add_argument('--logging_steps', type=int, default=500,
                              help="Log every X updates steps.")
          parser.add_argument('--save_steps', type=int, default=1500,
                              help="Save checkpoint every X updates steps.")
          parser.add_argument("--no_cuda", action='store_true',
                              help="Whether not to use CUDA when available")
          parser.add_argument("--do_train", action='store_true',
                              help="do training")
          parser.add_argument("--do_eval", action='store_true',
                              help="do eval")
          parser.add_argument("--do_test", action='store_true',
                              help="do test")
          parser.add_argument("--evaluate_during_training", action='store_true',
                              help="evaluate_during_training")
          parser.add_argument("--eval_all_checkpoints", action='store_true',
                              help="evaluate_during_training")
          parser.add_argument("--should_continue", action='store_true',
                              help="If we continue training from a checkpoint")
          parser.add_argument('--seed', type=int, default=42,
                              help="random seed for initialization")
          parser.add_argument("--head", dest="head", action="store_true")
          parser.add_argument("--no-head", dest="head", action="store_false")
          parser.set_defaults(head=True)
          parser.add_argument("--template", type=str, default=None)
          parser.add_argument("--local_rank", type=int, default=-1,
                              help="local_rank for distributed training on gpus")
          parser.add_argument('--fp16', action='store_true',
                              help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
          parser.add_argument('--fp16_opt_level', type=str, default='O1',
                              help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                   "See details at https://nvidia.github.io/apex/amp.html")


          self.parser = parser

     def parse(self):
          args = self.parser.parse_args()
          return args




MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig,XLMRobertaForMaskedLM, XLMRobertaTokenizer)
}   


def prepare(args):

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("--should_continue is true, but no checkpoint found in --output_dir")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device


    print("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    print("Training/evaluation parameters {}".format(args))

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def get_optimizer_scheduler(args, model, t_total):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total
    )


    return optimizer, scheduler


def prepare_for_training(args, model, train_dataloader):
    # total iteration and batch size
    if args.num_train_steps > 0:
        t_total = args.num_train_steps
        args.num_train_epochs = args.num_train_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.num_train_steps = t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    return args, model, optimizer, scheduler


def get_model_tokenizer(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )
    if args.head_type!="mlm":
        if args.model_name_or_path:
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir
            )
        else:
            print("Training new model from scratch")
            model = model_class(config=config)
    else:   #main model
        if args.model_name_or_path:
            model = XLMRobertaForMaskedLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir
            )
        else:
            print("Training new model from scratch")
            model = XLMRobertaForMaskedLM(config=config)
    

    model.to(args.device)
    
    if args.train_type=="headtune":
        for name, param in model.named_parameters():  #freeze all encoders
            if "classifier" not in name and "learned_embedding" not in name:
                param.requires_grad = False
    if args.train_type=="prompttune":   #only learn mask
        for name, param in model.named_parameters():
            if "learned_embedding" not in name:
                param.requires_grad = False
    return model, tokenizer, model_class, tokenizer_class, args


def get_training_info(dataloader, args):
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    return global_step, epochs_trained, steps_trained_in_current_epoch


def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, tr_loss, logging_loss, global_step,
                steps_trained_in_current_epoch, args,Verbalizer):
    """train one epoch"""
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    train_acc = 0

    model.train()

    for step, batch in enumerate(epoch_iterator):

        # Skip past any already trained steps if resuming training
        if steps_trained_in_current_epoch > 0:
            steps_trained_in_current_epoch -= 1
            continue

        src_ids = batch["src_ids"].to(args.device)
        src_mask = batch["src_mask"].to(args.device)
        label = batch["label"].to(args.device)


        if args.head_type!="mlm":
            outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)

            loss = outputs[0]
            
            train_acc += (outputs[1].argmax(1) == label.squeeze()).sum().item()
        else:
            if args.labelword_ensemble:
                
                outputs = model(input_ids=src_ids, attention_mask=src_mask)
            
                predict_score=outputs.logits[torch.arange(outputs.logits.size(0)), batch["mask_position"]]
                
                label0_logits=predict_score[:,Verbalizer["0"]].mean(dim=1)
                label1_logits=predict_score[:,Verbalizer["1"]].mean(dim=1)
                predict_label=torch.where(label0_logits > label1_logits,0,1)
                
                logits=torch.cat([label0_logits.unsqueeze(-1),label1_logits.unsqueeze(-1)],dim=-1)
                
                train_acc += (predict_label == label).sum().item()
                loss_fct = CrossEntropyLoss()
                loss=loss_fct(logits, label.view(-1))

            else:
                outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)
                loss = outputs[0]
                predict_score=outputs[1][torch.arange(outputs[1].size(0)), batch["mask_position"]]
                label=label[torch.arange(label.size(0)),batch["mask_position"]]
                negative=torch.tensor([Verbalizer["0"]]*label.size(0)).type_as(label)
                positive=torch.tensor([Verbalizer["1"]]*label.size(0)).type_as(label)
                predict_label=torch.where(predict_score[torch.arange(predict_score.size(0)),negative] > predict_score[torch.arange(predict_score.size(0)),positive],negative,positive)

                train_acc += (predict_label == label).sum().item()
                loss_fct = CrossEntropyLoss()
                loss=loss_fct(predict_score, label.view(-1))
                
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            # Log metrics
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                epoch_iterator.set_description(
                    "Training (lr=%2.5f) (loss=%2.5f)" % (scheduler.get_last_lr()[0], (tr_loss - logging_loss) / args.logging_steps)
                )

                logging_loss = tr_loss

        if args.num_train_steps > 0 and global_step > args.num_train_steps:
            epoch_iterator.close()
            break
    if args.data_type=="few":
        train_acc/=(args.n_shot*args.n_class)
    else:
        if args.src_domain=="all":
            train_acc/=6000
        else:
            train_acc/=2000
    

    print("train_acc:",train_acc)

    return model, optimizer, scheduler, global_step, tr_loss, logging_loss,train_acc


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    

    # Prepare dataloader
    train_dataloader, args = get_dataloader(train_dataset, tokenizer, args, split='train')
    args, model, optimizer, scheduler = prepare_for_training(args, model, train_dataloader)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    if args.local_rank in [-1, 0]:
        print("***** Running training *****")
        print("  Num examples = {}".format(len(train_dataset)))
        print("  Num Epochs = {}".format(args.num_train_epochs))
        print("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
        print("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(total_batch_size))
        print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
        print("  Total optimization steps = {}".format(args.num_train_steps))

    global_step, epochs_trained, steps_trained_in_current_epoch = get_training_info(train_dataloader, args)

    tr_loss, logging_loss = 0.0, 0.0

    DOMAINS = ('dvd', 'books', 'music')
    LANGUAGES=('en','fr','jp','de')
    besteval={}
    besttest={}
    bestepoch={}
    for language in LANGUAGES:
        for domain in DOMAINS:
            besteval[language+'_'+domain]=0
            besttest[language+'_'+domain]=0


    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    bestsum=0
    
    if args.labelword_ensemble:
        Verbalizer={"0":[],"1":[]}
        label0 = tokenizer.tokenize(args.label0)
        label1 = tokenizer.tokenize(args.label1)
        label0_ids=set(tokenizer.convert_tokens_to_ids(label0))  
        label1_ids=set(tokenizer.convert_tokens_to_ids(label1))
        input0_ids=list(label0_ids-label1_ids)
        input1_ids=list(label1_ids-label0_ids)
        Verbalizer["0"]+=input0_ids
        Verbalizer["1"]+=input1_ids

    else:
        Verbalizer={"0":tokenizer(args.label0)['input_ids'][1],"1":tokenizer(args.label1)['input_ids'][1]}
    

    #testresult=test(args, model, tokenizer,Verbalizer)
    #logger.info(testresult)
    #exit()
        
    for _ in train_iterator:

        model, optimizer, scheduler, global_step, tr_loss, logging_loss ,train_acc= train_epoch(model, tokenizer, optimizer,
                                                                                      scheduler, train_dataloader,
                                                                                      tr_loss, logging_loss,
                                                                                      global_step,
                                                                                      steps_trained_in_current_epoch,
                                                                                      args,Verbalizer)
        
        evalresult=evaluate(args, model, tokenizer,Verbalizer)
        
        if evalresult[args.src_language+'_'+args.src_domain]["eval_acc"]>=besteval[args.src_language+'_'+args.src_domain]: 
            #model.save_pretrained(args.src_domain+'-'+str(args.seed))
            
            for language in LANGUAGES:
                testresult=test(args, model, tokenizer,Verbalizer)                     
                besteval[language+'_'+args.src_domain]=evalresult[language+'_'+args.src_domain]["eval_acc"]
                besttest[language+'_'+args.src_domain]=testresult[language+'_'+args.src_domain]["eval_acc"]
                bestepoch[language+'_'+args.src_domain]=_
                
            bestsum=evalresult["sum_acc"]
        if args.local_rank in [-1, 0]:
                train_iterator.set_description(
                    "(train_loss=%2.5f) (train_acc=%2.5f)\nevalresult:%s\ntestresult:%s"%((tr_loss - logging_loss) / args.logging_steps,train_acc,str(evalresult),str(testresult)))
        if args.num_train_steps > 0 and global_step > args.num_train_steps:
            train_iterator.close()
            break
    print("*****************final result*****************")
    
    for language in LANGUAGES:
        print()
        print("language:",language,"domain:",args.src_domain)
        print("best eval result:",besteval[language+'_'+args.src_domain])
        print("test result:",besttest[language+'_'+args.src_domain])
        print("best epoch:",bestepoch[language+'_'+args.src_domain])

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, Verbalizer,prefix=""):
    DOMAINS = ('dvd', 'books', 'music')
    LANGUAGES=('en','fr','jp','de')
    result={"sum_acc":0}


    for language in LANGUAGES:
        for domain in DOMAINS:
            if args.src_domain=="all" or args.src_domain==domain:
                eval_dataset = SentimentDataset(args.max_len, language,domain,tokenizer,'valid',args.head_type,args,args.head, args.template)


                # Prepare dataloader
                eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='valid')

                # multi-gpu evaluate
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                # Eval!
                print("***** Running evaluation {} *****".format(prefix))
                print("  Num examples = {}".format(len(eval_dataset)))
                print("  Batch size = {}".format(args.eval_batch_size))
                eval_loss = 0.0
                nb_eval_steps = 0
                model.eval()
                eval_acc=0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    with torch.no_grad():
                        src_ids = batch["src_ids"].to(args.device)
                        src_mask = batch["src_mask"].to(args.device)
                        label = batch["label"].to(args.device)

                        if args.head_type!="mlm":
                            outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)

                            loss = outputs[0]
                            
                            eval_acc += (outputs[1].argmax(1) == label.squeeze()).sum().item()
                        else:
                            if args.labelword_ensemble:
                                
                                outputs = model(input_ids=src_ids, attention_mask=src_mask)

                                predict_score=outputs.logits[torch.arange(outputs.logits.size(0)), batch["mask_position"]]
                                label0_logits=predict_score[:,Verbalizer["0"]].mean(dim=1)
                                label1_logits=predict_score[:,Verbalizer["1"]].mean(dim=1)
                                predict_label=torch.where(label0_logits > label1_logits,0,1)
                                logits=torch.cat([label0_logits.unsqueeze(-1),label1_logits.unsqueeze(-1)],dim=-1)
                                eval_acc += (predict_label == label).sum().item()
                                loss_fct = CrossEntropyLoss()
                                loss=loss_fct(logits, label.view(-1))
                            else:
                                outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)
                                loss = outputs[0]
                                predict_score=outputs[1][torch.arange(outputs[1].size(0)), batch["mask_position"]]
                                label=label[torch.arange(label.size(0)),batch["mask_position"]]
                                negative=torch.tensor([Verbalizer["0"]]*label.size(0)).type_as(label)
                                positive=torch.tensor([Verbalizer["1"]]*label.size(0)).type_as(label)
                                predict_label=torch.where(predict_score[torch.arange(predict_score.size(0)),negative] > predict_score[torch.arange(predict_score.size(0)),positive],negative,positive)

                                eval_acc += (predict_label == label).sum().item()
                                loss_fct = CrossEntropyLoss()
                                loss=loss_fct(predict_score, label.view(-1))

                    eval_loss += loss.mean().item()
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps

                eval_acc/=(args.n_shot*args.n_class)

                result[language+'_'+domain] = {"eval_acc": eval_acc,"loss":eval_loss}
                result["sum_acc"]+=eval_acc
                print("***** Eval results ",language+'_'+domain,"{} *****".format(prefix))
                for key in sorted(result.keys()):
                    print("  {} = {}".format(key, str(result[key])))
            
    return result

def test(args, model, tokenizer, Verbalizer,prefix=""):
    DOMAINS = ('dvd', 'books', 'music')
    LANGUAGES=['en','de','fr','jp']
    label0_sum=0
    label1_sum=0

    result={}
    for language in LANGUAGES:
        for domain in DOMAINS:
            if args.src_domain=="all" or args.src_domain==domain:
                eval_dataset = SentimentDataset(args.max_len,language,domain,tokenizer,'test',args.head_type,args,args.head, args.template)

                # Prepare dataloader
                eval_dataloader, args = get_dataloader(eval_dataset, tokenizer, args, split='valid')

                # multi-gpu evaluate
                if args.n_gpu > 1:
                    model = torch.nn.DataParallel(model)

                # Eval!
                print("***** Running evaluation {} *****".format(prefix))
                print("  Num examples = {}".format(len(eval_dataset)))
                print("  Batch size = {}".format(args.eval_batch_size))
                eval_loss = 0.0
                nb_eval_steps = 0
                model.eval()
                eval_acc=0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    with torch.no_grad():
                        src_ids = batch["src_ids"].to(args.device)
                        src_mask = batch["src_mask"].to(args.device)
                        label = batch["label"].to(args.device)

                        if args.head_type!="mlm":
                            outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)
                            loss = outputs[0]
                            eval_acc += (outputs[1].argmax(1) == label.squeeze()).sum().item()

                        else:
                            if args.labelword_ensemble:                               
                                outputs = model(input_ids=src_ids, attention_mask=src_mask)                                
                                predict_score=outputs.logits[torch.arange(outputs.logits.size(0)), batch["mask_position"]]                                
                                label0_logits=predict_score[:,Verbalizer["0"]].mean(dim=1)                               
                                label1_logits=predict_score[:,Verbalizer["1"]].mean(dim=1)                                
                                predict_label=torch.where(label0_logits > label1_logits,0,1)                                
                                logits=torch.cat([label0_logits.unsqueeze(-1),label1_logits.unsqueeze(-1)],dim=-1)
                                eval_acc += (predict_label == label).sum().item()
                                loss_fct = CrossEntropyLoss()
                                loss=loss_fct(logits, label.view(-1))
                                
                            else:
                                outputs = model(input_ids=src_ids, attention_mask=src_mask, labels=label)
                                loss = outputs[0]                               
                                predict_score=outputs[1][torch.arange(outputs[1].size(0)), batch["mask_position"]]
                                label=label[torch.arange(label.size(0)),batch["mask_position"]]
                                negative=torch.tensor([Verbalizer["0"]]*label.size(0)).type_as(label)
                                positive=torch.tensor([Verbalizer["1"]]*label.size(0)).type_as(label)
                                predict_label=torch.where(predict_score[torch.arange(predict_score.size(0)),negative] > predict_score[torch.arange(predict_score.size(0)),positive],negative,positive)

                                eval_acc += (predict_label == label).sum().item()
                                loss_fct = CrossEntropyLoss()
                                loss=loss_fct(predict_score, label.view(-1))

                        eval_loss += loss.mean().item()
                        nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_acc/=2000

                result[language+'_'+domain]= {"eval_acc": eval_acc,"loss":eval_loss}
                print("***** Test results ",language+'_'+domain,"{} *****".format(prefix))
                for key in sorted(result.keys()):
                    print("  {} = {}".format(key, str(result[key])))

    return result


def main():
    args = ArgsParser().parse()
    prepare(args)

    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # if not the first process, do not load pretrained model & vocab

    model, tokenizer, model_class, tokenizer_class, args = get_model_tokenizer(args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # finish barrier, when first process has loaded pretrained model & vocab

    

    print("Training/evaluation parameters {}".format(args))

    # Training
    if args.do_train:
        train_dataset = SentimentDataset(args.max_len, args.src_language,args.src_domain,tokenizer,'train',args.head_type,args,args.head, args.template,args.data_type)
        global_step, train_loss = train(args, train_dataset, model, tokenizer)
        print(" global_step = {}, average loss = {}".format(global_step, train_loss))


if __name__ == "__main__":
    main()
