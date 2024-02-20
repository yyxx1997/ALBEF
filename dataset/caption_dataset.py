import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


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
            self.img2txt[img_id] = set()
            for caption in ann['caption']:
                caption_clean = pre_caption(caption, self.max_words)
                if caption_clean not in self.text:
                    self.txt2img[cnt] = set()
                    txt_id = cnt
                    self.text.append(caption_clean)
                    self.origin_text.append(caption)
                    cnt += 1
                else:
                    txt_id = self.text.index(caption_clean)
                self.img2txt[img_id].add(txt_id)
                self.txt2img[txt_id].add(img_id)  
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    

        image_name = self.image[index]
        image_path = os.path.join(self.image_root, image_name)        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
class re_entail_train_dataset(Dataset):
    def __init__(self, ann_file, entailments, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.entailments = json.load(open(entailments,'r'))
        self.img2txt_entail = {}
        self.txt2img_entail = {}
        self.goldens = {}
        for image_path, dct in self.entailments.items():
            goldens = dct['goldens']
            self.goldens[image_path] = goldens
            entailments = dct['entailments']
            if entailments:
                self.img2txt_entail[image_path] = entailments
            for txt in goldens:
                txt = pre_caption(txt,max_words=512)
                self.txt2img_entail[txt] = self.txt2img_entail.get(txt,[])+[image_path]
            for txt in entailments:
                txt = pre_caption(txt,max_words=512)
                self.txt2img_entail[txt] = self.txt2img_entail.get(txt,[])+[image_path]  
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
        image_name = ann['image']
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 
        if image_name in self.img2txt_entail.keys():
            entail_pools = self.img2txt_entail[image_name]
            random_entail = random.sample(entail_pools,1)[0]
        elif image_name in self.goldens.keys():
            random_entail = random.sample(self.goldens[image_name],1)[0]
        else:
            random_entail = caption
        return image, caption, self.img_ids[ann['image_id']], random_entail

class re_entail_lr_train_dataset(Dataset):
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
        gold = ann['gold'] 

        return image, caption, self.img_ids[ann['image_id']], gold

class re_entail_lr_split_train_dataset(Dataset):
    def __init__(self, ann_file, entailments, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.entailments = json.load(open(entailments,'r'))
        self.img2txt_entail = {}
        self.goldens = {}
        for image_path, dct in self.entailments.items():
            goldens = dct['goldens']
            self.goldens[image_path] = goldens
            entailments = dct['entailments']
            if entailments:
                self.img2txt_entail[image_path] = entailments
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
        ...
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_name = ann['image']
        image_path = os.path.join(self.image_root,image_name)        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 
        if image_name in self.img2txt_entail.keys():
            entail_pools = self.img2txt_entail[image_name]
            random_entail = random.sample(entail_pools,1)[0]
        elif image_name in self.goldens.keys():
            random_entail = random.sample(self.goldens[image_name],1)[0]
        else:
            random_entail = caption
        random_entail = pre_caption(random_entail, self.max_words)
        idx = self.img_ids[ann['image_id']]
        return image, caption, idx, random_entail