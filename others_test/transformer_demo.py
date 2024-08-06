import math
import torch
import torch.nn as nn

"""
1. input embedding
    input-> tensor
2. position encoding
token embedding & position embedding

3. encoder
  (1) multi-head attention
    1) self-attention
  (2) add & norm
  (3) feed forward

4. decoder
"""

src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
             '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}

# Transformer Parameters
d_model = 512  # Embedding Size

sentences = [
    # 中文和英语的单词个数不要求相同
    # enc_input                dec_input           dec_output
    ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
    ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
    ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
]

def transfer_token_to_tensor(input_sentences):
    enc_inputs = []
    for i in range(len(input_sentences)):
        enc_input = [[src_vocab[n] for n in input_sentences[i][0].split()]]

        enc_inputs.extend(enc_input)

    return torch.LongTensor(enc_inputs)       

"""
position encoding shape:
  max_len x d_model
  max_len: max token number
"""
class PositionEncoding(nn.Module):
  def __init__(self, d_model, max_len=50000):
    super(PositionEncoding, self).__init__()

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term =torch.exp(torch.arange(
      0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)  
    pe = pe.unsqueeze(0).transpose(0, 1) #[seq_len, batch_size, d_model]
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
    x: [seq_len, batch_size, d_model]
    """
    x = x + self.pe[:x.size(0), :]
    return x

"""

"""
#class ScaledDotProductAttention(nn.Module):
   

if __name__=="__main__":
    enc_inputs = transfer_token_to_tensor(sentences)
    print(enc_inputs)
    