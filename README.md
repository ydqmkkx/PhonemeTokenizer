# **PhonemeTokenizer**
This phoneme tokenizer converts text to phoneme (ARPAbet) tokens.\
The G2P module derives from https://github.com/Kyubyong/g2p.\
The function design is inspired by the tokenizer implementations from Hugging Face.

The tokenization method derives from our work (in press)：
1. For the phonemes of a word, we append the prefix ## to all subsequent phonemes.
2. To better align with TTS tasks, we retain only the first punctuation for consecutive punctuation marks.

## Install
Requirements can be found in https://github.com/Kyubyong/g2p, `torch` is also needed:
```
inflect>=0.3.1
nltk>=3.2.4
numpy>=1.13.1
torch
```

## Usage
`return_tensors` (default: None): now we only support "pt".
```python
from PhonemeTokenizer import PhonemeTokenizer
p_tn = PhonemeTokenizer()

p_tn("Hello, world!?")
# {'input_ids': [1, 20, 46, 64, 68, 94, 40, 55, 64, 52, 83, 2],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

p_tn(["Hello, world!?", "α"], return_tensors="pt")
# {'input_ids': tensor([[ 1, 20, 46, 64, 68, 94, 40, 55, 64, 52, 83,  2],
#                       [ 1,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0]]),
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                           [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
```

### .g2p(text, prefix=True, stress=False)
`prefix`: whether to use continuing prefix ##.\
`stress`: whether to include stress markers.\
The output are aligned sequences of words and phonemes.
```python
p_tn.g2p("Hello, world!?")
# (['hello', ',', 'world', '!'], ['HH ##AH ##L ##OW', ',', 'W ##ER ##L ##D', '!'])

p_tn.g2p("Hello, world!?", prefix=False, stress=True)
# (['hello', ',', 'world', '!'], ['HH AH0 L OW1', ',', 'W ER1 L D', '!'])
```

### Others
You can obtain the phoneme vocabulary through `p_tn.get_vocab()`
```python
tokens = p_tn.tokenize("Hello, world!?")
# tokens: ['HH', '##AH', '##L', '##OW', ',', 'W', '##ER', '##L', '##D', '!']

token_ids = p_tn.convert_tokens_to_ids(tokens)
# token_ids: [20, 46, 64, 68, 94, 40, 55, 64, 52, 83]

tokens = p_tn.convert_ids_to_tokens(token_ids)
# tokens: ['HH', '##AH', '##L', '##OW', ',', 'W', '##ER', '##L', '##D', '!']
```