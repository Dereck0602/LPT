import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler,WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import copy
import logging
import sys

logger = logging.getLogger(__name__)
     
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
logger.setLevel(logging.INFO)

class SentimentDataset(Dataset):

    def __init__(self,max_len,language,domain, tokenizer,train,head_type,args,head, template,data_type="all"):
        self.max_len=max_len
        self.tokenizer = tokenizer
        self.data =[]
        self.label=[] 
        self.mask_position=[]
        self.head_type=head_type
        self.data_type=data_type
        self.labelword_ensemble=args.labelword_ensemble
        self.head = head
        self.template = template
        self.load_data(language,domain,train,args)

    def __getitem__(self, idx):

        text = self.data[idx]
        label= self.label[idx]
        
        if self.head_type!="mlm":

            return {"src_ids": text["input_ids"],
                        "src_mask":text["attention_mask"],
                        "label":label}
        else:  #used
            mask_position= self.mask_position[idx]

            if self.labelword_ensemble:

                return {"src_ids": text["input_ids"],
                            "src_mask":text["attention_mask"],
                            "label":label,
                            "mask_position":mask_position}
            else:

                return {"src_ids": text["input_ids"],
                            "src_mask":text["attention_mask"],
                            "label":label["input_ids"],
                            "mask_position":mask_position}

    def __len__(self):
        return len(self.data)
    
    def load_data(self,language,domain,train,args):
        DOMAINS = ('dvd', 'books', 'music')
        Verbalizer={"0":self.tokenizer(args.label0)['input_ids'][1],"1":self.tokenizer(args.label1)['input_ids'][1]}
        if self.head_type!="mlm":
            for tgt_domain in DOMAINS:
                if domain=="all" or domain==tgt_domain:
                    with open ('./multiamazone/'+language+'/'+tgt_domain+'/32shot/'+train+'.txt', 'r') as F:
                        for line in F:
                            text=self.tokenizer(line[2:], pad_to_max_length=True, truncation=True,max_length=self.max_len)
                            self.data.append(text)
                            self.label.append([int(line[0])])

        else:
            if args.labelword_ensemble:
                for tgt_domain in DOMAINS:
                    if domain=="all" or domain==tgt_domain:
                        with open ('./multiamazone/'+language+'/'+tgt_domain+'/32shot/'+train+'.txt', 'r') as F:
                            for line in F:
                                text=self.tokenizer(line[2:], pad_to_max_length=True, truncation=True,max_length=self.max_len)
                                
                                eos_position=text["input_ids"].index(self.tokenizer.eos_token_id)  #tokenizer.eos_token_id 2
                                
                                template = self.template
                                
                                if template != None:
                                    if self.head == 0: # tail
                                        template_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(template))
                                        masknumber = len(template_ids)
                                        if eos_position==args.max_len-1:
                                            text["input_ids"][eos_position-masknumber:eos_position]=template_ids
                                        elif eos_position<=args.max_len-1-masknumber:
                                            text["input_ids"][eos_position:eos_position+masknumber]=template_ids
                                            text["input_ids"][eos_position+masknumber]=self.tokenizer.eos_token_id
                                            text["attention_mask"][eos_position+1:eos_position+masknumber+1]=[1 for i in range(masknumber)]
                                        else:
                                            text["input_ids"][args.max_len-1-masknumber:args.max_len-1]=template_ids
                                            text["input_ids"][args.max_len-1]=self.tokenizer.eos_token_id
                                            text["attention_mask"]=[1 for i in range(args.max_len)]
                                    
                                    else:                                
                                        template_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(template))
                                        for i in range(1,len(template_ids)+1):
                                            text["input_ids"].insert(i,template_ids[i-1])
                                            text["attention_mask"].insert(i,1)
                                        text["input_ids"] = text["input_ids"][:args.max_len]
                                        text["attention_mask"] = text["attention_mask"][:args.max_len]
                                        if text["input_ids"][-1]!=(1 or 2):
                                            text["input_ids"][-1]=2
                                    self.mask_position.append(text["input_ids"].index(self.tokenizer.mask_token_id))
                                else:
                                    self.mask_position.append(text["input_ids"].index(self.tokenizer.cls_token_id))
                                labels=int(line[0])
                                self.data.append(text)
                                self.label.append(labels)
                                
                                
                                

class WikiDataset(Dataset):

    def __init__(self,max_len,language,tokenizer,train,head_type,args,head, template,data_type="all"):
        self.max_len=max_len
        self.tokenizer = tokenizer
        self.data =[]
        self.label=[] 
        self.mask_position=[]
        self.head_type=head_type
        self.data_type=data_type
        self.head = head
        self.template = template
        self.load_data(language,train,args)

    def __getitem__(self, idx):

        text = self.data[idx]
        label= self.label[idx]
        
        if self.head_type!="mlm":

            return {"src_ids": text["input_ids"],
                        "src_mask":text["attention_mask"],
                        "label":label}
        else:  #used
            mask_position= self.mask_position[idx]

            return {"src_ids": text["input_ids"],
                        "src_mask":text["attention_mask"],
                        "label":label,
                        "mask_position":mask_position}

        

    def __len__(self):
        return len(self.data)
    
    def load_data(self,language,train,args):
        if self.head_type!="mlm":
            with open ('./parallel/'+language+'.txt', 'r') as F:
                for line in F:
                    text=self.tokenizer(line, pad_to_max_length=True, truncation=True,max_length=self.max_len)
                    self.data.append(text)
                    self.label.append(0)

        else:
            with open ('./parallel/'+language+'.txt', 'r') as F:
                for line in F:
                    text=self.tokenizer(line, pad_to_max_length=True, truncation=True,max_length=self.max_len)
                    
                    eos_position=text["input_ids"].index(self.tokenizer.eos_token_id)  #tokenizer.eos_token_id 2
                    
                    template = self.template
                                
                    if template != None:
                        if self.head == 0: # tail
                            template_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(template))
                            masknumber = len(template_ids)
                            if eos_position==args.max_len-1:
                                text["input_ids"][eos_position-masknumber:eos_position]=template_ids
                            elif eos_position<=args.max_len-1-masknumber:
                                text["input_ids"][eos_position:eos_position+masknumber]=template_ids
                                text["input_ids"][eos_position+masknumber]=self.tokenizer.eos_token_id
                                text["attention_mask"][eos_position+1:eos_position+masknumber+1]=[1 for i in range(masknumber)]
                            else:
                                text["input_ids"][args.max_len-1-masknumber:args.max_len-1]=template_ids
                                text["input_ids"][args.max_len-1]=self.tokenizer.eos_token_id
                                text["attention_mask"]=[1 for i in range(args.max_len)]
                        
                        else:                                
                            template_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(template))
                            for i in range(1,len(template_ids)+1):
                                text["input_ids"].insert(i,template_ids[i-1])
                                text["attention_mask"].insert(i,1)
                            text["input_ids"] = text["input_ids"][:args.max_len]
                            text["attention_mask"] = text["attention_mask"][:args.max_len]
                            if text["input_ids"][-1]!=(1 or 2):
                                text["input_ids"][-1]=2
                        self.mask_position.append(text["input_ids"].index(self.tokenizer.mask_token_id))
                    else:
                        self.mask_position.append(text["input_ids"].index(self.tokenizer.cls_token_id))
                    
                    self.data.append(text)
                    self.label.append(0)
                                
        



def get_dataloader(dataset, tokenizer, args, split='train'):

    def Sentimentcollate_fn(batch):
        """
        Modify target_id as label
        """

        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        label = torch.tensor([example['label'] for example in batch], dtype=torch.long)

        
        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "label":label}
    
    def MASKSentimentcollate_fn(batch):
        """
        Modify target_id as label
        """

        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        label = torch.tensor([example['label'] for example in batch], dtype=torch.long)
        mask_position=torch.tensor([example['mask_position'] for example in batch], dtype=torch.long)

        
        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "label":label,
                "mask_position":mask_position}

    def MLMSentimentcollate_fn(batch):
        """
        Modify target_id as label
        """

        src_ids = torch.tensor([example['src_ids'] for example in batch], dtype=torch.long)
        src_mask = torch.tensor([example['src_mask'] for example in batch], dtype=torch.long)
        label = torch.tensor([example['label'] for example in batch], dtype=torch.long)
        label[label[:, :] == 1] = -100
        mask_position=torch.tensor([example['mask_position'] for example in batch], dtype=torch.long)
        
        return {"src_ids": src_ids,
                "src_mask":src_mask,
                "label":label,
                "mask_position":mask_position}

    if split == 'train':
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        batch_size = args.train_batch_size
        sampler = RandomSampler(dataset if args.local_rank == -1 else DistributedSampler(dataset)) # SequentialSampler(dataset)
    elif split == 'valid':
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)    
        batch_size = args.eval_batch_size
        sampler = SequentialSampler(dataset)

    if args.head_type!="mlm" :
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=Sentimentcollate_fn)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=MASKSentimentcollate_fn)  #used
        


    return dataloader, args