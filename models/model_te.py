import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset

class BertTEModel(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-uncased'):
        """
        Args: 
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        super(BertTEModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        self.classifier = nn.Linear(768, class_size)
        self.sm=nn.Softmax(dim=-1)

    def forward(self, input_ids, input_tyi, input_attn_mask):
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        categories_numberic = self.sm(self.classifier(output.pooler_output))
        return categories_numberic

class BertTEEvalDataset(Dataset):
    def __init__(self, pairs,images,texts):
        self.pairs = pairs
        self.data_size = len(pairs)
        self.images=images
        self.texts=texts

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        pre_id,hypo_id = self.pairs[index]
        premise=self.texts[pre_id]
        hyposis=self.texts[hypo_id]
        return premise,hyposis,index