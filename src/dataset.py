import config
import torch

class BertDataset():
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = config.TOKENIZERS
        self.max_len = config.MAX_LENGTH

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        review = " ".join(review.split())

        output = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids = output['input_ids']
        mask = output['attention_mask']
        token_type_ids = output['token_type_ids']

        padding_len = self.max_len - len(ids)

        ids = ids + ( [0]*padding_len )
        mask = mask + ( [0]*padding_len )
        token_type_ids = token_type_ids + ( [0]*padding_len )

        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
            'target' : torch.tensor(self.targets[item], dtype=torch.float)
        }
