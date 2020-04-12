import config

# Transformer and Torch modules
import transformers
import torch.nn as nn

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class BertBaseUncase(nn.Module):
    def __init__(self):
        super(BertBaseUncase, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_UNCASED_PATH
        )
        if config.FEATURE_EXTRACTING:
            set_parameter_requires_grad(self.bert)

        self.drop_out = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        bert_drop = self.drop_out(out2)
        output = self.out(bert_drop)
        return output