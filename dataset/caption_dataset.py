import json
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text


def mixgen_batch(image, text, num, lam=0.5):
    batch_size = image.size()[0]
    index = np.random.permutation(batch_size)
    for i in range(batch_size):
        if i >= num: break
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[index[i],:]
        # text concat
        text[i] = text[i] + " " + text[index[i]]
    return image, text

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]

    def collate_fn(self, batchs):
        images = []
        texts = []
        img_ids = []
        for image, text, idx in batchs:
            images.append(image)
            texts.append(text)
            img_ids.append(idx)
        images = torch.stack(images)
        img_ids = torch.tensor(img_ids)
        return images, texts, img_ids
    
class re_train_dataset_mixgen(Dataset):
    def __init__(self, ann_file, transform, image_root, mix_rate=0.25, mix_lam=0.5, mode="mixgen", max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1  

        self.mix_rate = mix_rate
        self.mix_lam = mix_lam
        self.mix_mode = mode
        self.mix_len = self.__len__()
        self.mixgen_order = np.random.choice(self.mix_len, self.mix_len, replace=False)
        self.cur = 0  
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    def collate_fn(self, batchs):
        # TODO pick samples according to cosine sim
        # 1. random mixgen instead in batch √
        # 2. pick samples according to cosine sim
        # 3. filter by entailment model
        N = int(self.mix_rate * len(batchs))
        mixgen_size = min(len(batchs), N)
        images = []
        texts = []
        img_ids = []
        if self.mix_mode == 'mixgen_random':
            for i, (image, text, idx) in enumerate(batchs):
                if i < mixgen_size:
                    self.cur = (self.cur + 1) % self.__len__()
                    image_cand, text_cand, _ = self.__getitem__(self.mixgen_order[self.cur])
                    # image mixup
                    image = self.mix_lam * image + (1 - self.mix_lam) * image_cand
                    # text concat
                    text = text + " " + text_cand
                images.append(image)
                texts.append(text)
                img_ids.append(idx)
            images = torch.stack(images)
        else:
            for image, text, idx in batchs:
                images.append(image)
                texts.append(text)
                img_ids.append(idx)
            images = torch.stack(images)
            if self.mix_mode == "mixgen":
                mixgen(images, texts, N, self.mix_lam)
            elif self.mix_mode == "mixgen_batch":
                mixgen_batch(images, texts, N, self.mix_lam)
        img_ids = torch.tensor(img_ids)
        return images, texts, img_ids

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.origin_text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        cnt = 0 
        for img_id, ann in enumerate(self.ann):
            image_name=ann['image']
            self.image.append(image_name)
            self.img2txt[img_id] = []
            for caption in ann['caption']:
                caption_clean = pre_caption(caption, self.max_words)
                if caption_clean not in self.text:
                    txt_id = cnt
                    self.txt2img[txt_id] = []
                    self.text.append(caption_clean)
                    self.origin_text.append(caption)
                    cnt += 1
                else:
                    txt_id = self.text.index(caption_clean)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id].append(img_id)  
        assert len(self.text) == len(self.origin_text), "Error in text processing!!"
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
            

def read_jsonl(filename):
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            line = line.strip()
            dct = json.loads(line)
            yield dct      

class re_train_dataset_cscore(Dataset):
    def __init__(self, ann_file, clip_score_file, transform, image_root, mix_rate=0.25, mix_lam=0.5, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1  

        self.mix_rate = mix_rate
        self.mix_lam = mix_lam

        self.text = []
        self.txt2img = {} 
        self.txt2num = {}
        cnt = 0
        for ann in tqdm(self.ann, total=len(self.ann), desc="Loading dataset..."):
            image_name = ann['image']
            caption = ann['caption']
            if caption not in self.txt2img:
                self.txt2img[cnt] = image_name
                self.txt2num[caption] = cnt
                self.text.append(caption)
                cnt += 1

        clip_score = read_jsonl(clip_score_file)
        self.topk_texts = {}
        self.topk_scores = {}

        for dct in tqdm(clip_score, desc="Loading clip score vocab..."):
            # if len(self.topk_texts) > 10000: break
            image_name = dct['image']
            topk_scores = dct['topk_scores']
            self.topk_texts[image_name] = []
            scores = []
            cnt = 0
            for txt, score in topk_scores.items():
                if cnt >= 32:
                    break
                idx = self.txt2num.get(txt, -1)
                if idx >= 0 and score > 0:
                    self.topk_texts[image_name].append(idx)
                    scores.append(score)
                    cnt += 1
            scores = np.array(scores, dtype=np.float16)
            self.topk_scores[image_name] = scores / scores.sum()
        
    def __len__(self):
        return len(self.ann)
    
    def get_similar_alter(self, query_image):
        if query_image not in self.topk_texts:
            return None, None
        txt_ids = self.topk_texts[query_image]
        scores = self.topk_scores[query_image]
        cnt = len(txt_ids)
        while cnt:
            idx = np.random.choice(a=len(scores), size=1, replace=False, p=scores).item()
            txt_id = txt_ids[idx]
            image_name = self.txt2img[txt_id]
            if image_name != query_image:
                break
            cnt -= 1
        else:
            return None, None

        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        caption = pre_caption(self.text[txt_id], self.max_words)

        return image, caption

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_name = ann['image']
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        idx = self.img_ids[ann['image_id']]
        caption = pre_caption(ann['caption'], self.max_words) 
        image_alt, caption_alt = self.get_similar_alter(image_name)

        return image, caption, idx, image_alt, caption_alt 
    
    def collate_fn(self, batchs):
        # TODO pick samples according to cosine sim
        # 1. random mixgen instead in batch √
        # 2. pick samples according to cosine sim √
        # 3. filter by entailment model
        N = int(self.mix_rate * len(batchs))
        mixgen_size = min(len(batchs), N)
        images = []
        texts = []
        img_ids = []
        i = 0
        for image, text, idx, image_alt, caption_alt in batchs:
            if i < mixgen_size and image_alt is not None and caption_alt is not None:
                # image mixup
                image = self.mix_lam * image + (1 - self.mix_lam) * image_alt
                # text concat
                text = text + " " + caption_alt
                i += 1
            images.append(image)
            texts.append(text)
            img_ids.append(idx)
        images = torch.stack(images)
        img_ids = torch.tensor(img_ids)
        return images, texts, img_ids