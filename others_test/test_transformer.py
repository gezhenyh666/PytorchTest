import torch
import torch.nn as nn
import torch.utils.data as Data

sentences = [
        # enc_input      
        ['ich mochte ein bier P'],
        ['ich mochte ein cola P']
]

# Padding Should be Zero
source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
source_vocab_sizehg  = len(source_vocab)
source_len = 5 # max length of input sequence, 其实就是最长的那句话的token数


def make_data(sentences):
  encoder_inputs = []
  for i in range(len(sentences)):
    encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
    encoder_inputs.append(encoder_input)
  return torch.LongTensor(encoder_inputs)


# 使用Dataset加载数据
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs):
        super(MyDataSet,self).__init__()
        self.enc_inputs = enc_inputs
        
    def __len__(self):
        # 句子数量
        return self.enc_inputs.shape[0] 
    
    # 根据idx返回的是一组 enc_input
    def __getitem__(self, idx):
        return self.enc_inputs[idx]
 

d_model = 512 #一个词的向量长度
#FFN的隐藏层神经元个数
d_ff = 2048
#分头后的q、k、v词向量长度
d_k = d_v = 64

# Encoder Layer 和 Decoder Layer的个数
n_layers = 6

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # dropout是原文的0.1，max_len原文没找到
        '''max_len是假设的一个句子最多包含5000个token'''
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 开始位置编码部分,先生成一个max_len * d_model 的矩阵，即5000 * 512
        # 5000是一个句子中最多的token数，512是一个token用多长的向量来表示，5000*512这个矩阵用于表示一个句子的信息
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # pos：[max_len,1],即[5000,1]
        # 先把括号内的分式求出来,pos是[5000,1],分母是[256],通过广播机制相乘后是[5000,256]
        div_term = pos / pow(10000.0,torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)
        # 一个句子要做一次pe，一个batch中会有多个句子，所以增加一维用来和输入的一个batch的数据相加时做广播
        pe = pe.unsqueeze(0) # [5000,512] -> [1,5000,512] 
        # 将pe作为固定参数保存到缓冲区，不会被更新
        self.register_buffer('pe', pe)
        
        
    def forward(self, x):
        '''x: [batch_size, seq_len, d_model]'''
        # 5000是我们预定义的最大的seq_len，就是说我们把最多的情况pe都算好了，用的时候用多少就取多少
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) # return: [batch_size, seq_len, d_model], 和输入的形状相同

if __name__=="__main__":
  enc_inputs = make_data(sentences)
  # 构建DataLoader
  loader = Data.DataLoader(dataset=MyDataSet(enc_inputs),batch_size=2,shuffle=True)
  print(' enc_inputs: \n', enc_inputs)  # enc_inputs: [2,5]
  transformer_input = PositionalEncoding(d_model)