import torch
from torch.nn.utils.rnn import pad_sequence
from . import vocab
from .g2p import G2p
g2p = G2p()


class PhonemeTokenizer:
    def __init__(self):
        super().__init__()
        self.g2p = G2p()
        self.vocab = vocab._vocab
        self.reversed_vocab = vocab._reversed_vocab
    
    def tokenize(self, text):
        p_tokens = " ".join(self.g2p(text)[1]).split()
        return p_tokens
    
    def convert_tokens_to_ids(self, p_tokens):
        p_token_ids = list(map(lambda p: self.vocab.get(p, 4), p_tokens))
        return p_token_ids

    def convert_ids_to_tokens(self, p_token_ids):
        p_tokens = list(map(lambda p: self.reversed_vocab.get(p, "[UNK]"), p_token_ids))
        return p_tokens
    
    def _tokenize(self, text):
        p_tokens = self.tokenize(text)
        p_token_ids = self.convert_tokens_to_ids(p_tokens)
        return [1] + p_token_ids + [2] # [self.vocab["CLS"]] + p_token_ids + [self.vocab["SPE"]]

    def __call__(self, text, return_tensors=None):
        if isinstance(text, str):
            p_token_ids = self._tokenize(text)
            attn_mask = [1] * len(p_token_ids)
            if not return_tensors:
                return {'input_ids': p_token_ids, 'attention_mask': attn_mask}
            elif return_tensors == "pt":
                return {'input_ids': torch.tensor([p_token_ids]), 'attention_mask': torch.tensor([attn_mask])}
            else:
                raise ValueError("We only support 'pt' now.")

        elif isinstance(text, list):
            p_tokens_list = []
            for t in text:
                p_token_ids = torch.tensor(self._tokenize(t))
                p_tokens_list.append(p_token_ids)
            p_tokens_list = pad_sequence(p_tokens_list, batch_first=True)
            attn_mask = p_tokens_list.ne(0).long()
            if return_tensors == "pt":
                return {'input_ids': p_tokens_list, 'attention_mask': attn_mask}
            elif not return_tensors:
                return {'input_ids': p_tokens_list.tolist(), 'attention_mask': attn_mask.tolist()}
            else:
                raise ValueError("We only support 'pt' now.")
        
        else:
            raise ValueError("Text input must be of type `str` (single example), `List[str]`.")
