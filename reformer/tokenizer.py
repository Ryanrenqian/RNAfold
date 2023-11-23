class RNATokenizer:
    """RNA Tokenizer."""

    def __init__(self):
        self.stoi = {
            'A':0,
            'C':1,
            'G':2,
            'U':3,
            'P':4, # pad
            'M':5, # mask
            }
        self.itos = {
            value: key for key, value in self.stoi.items()
        }
        self.vocab_size = 6
        assert self.vocab_size == len(self.stoi.keys())

    @property
    def mask_token(self):
        return 'M'
    
    @property
    def PAD_token(self):
        return 'M'
    
    @property
    def pad_token_id(self):
        return self.stoi['P']

    @property
    def mask_token_id(self):
        return self.stoi['M']


    def tokenize(self, text):
        """Tokenize the text, converting it into a list of tokens (chars in our case). """
        output = [c for c in text]
        return output

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        unknown_id = self.stoi['U']
        return [self.stoi.get(tok, unknown_id) for tok in tokens]

    def encode(self, text):
        """Convert the string text to a list of tokenized ids.
        """
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, ids):
        """Converts a sequence of ids to a string"""
        if not isinstance(ids, list):
            ids = [ids]  # at least a length one list
        output = [self.itos.get(item, 'U') for item in ids]
        output = ''.join(output)
        return output

if __name__ =='__main__':
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers import CharBPETokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import Lowercase, NFKC, Sequence
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer



    # 4、添加解码器，将token令牌化的输入恢复为原始的输入
    tokenizer = CharBPETokenizer()
    # 6、开始训练我们的语料
    tokenizer.train(files=["../datas/vocabulary.txt"])
    special_tokens = ["[PAD]", "[SEP]", "[MASK]"]  
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASk]"
    # 最终得到该语料的Tokenizer，查看下词汇大小
    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    # 保存训练的tokenizer
    tokenizer.save('./my_tokens')
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./my_tokens")
    print(tokenizer)

