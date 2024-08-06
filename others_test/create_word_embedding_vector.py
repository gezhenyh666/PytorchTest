import torch
import torch.nn as nn
from torchtext.vocab import vocab

from collections import Counter

text = "hello world hello" # 假设我们有一个文本
counter = Counter(text.split()) # 统计每个单词的出现次数
print(type(counter))
v1 = vocab(counter) # 创建词汇表
print(type(v1))

# 构建嵌入层
vocab_size = 1000
embedding_dim = 100
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
print(embedding_layer)

# 假设我们有一个输入单词列表，每个单词都是从词汇表中随机选择的
input_words = ["hello", "world", "this", "is", "a", "test"]
for word in input_words:
    print(word)
# 将单词转换为索引列表
#word_indexes = [vocab.index(word) for word in input_words]
word_indexes = [v1.stoi[word] for word in input_words]
#print(word_indexes)
# 将索引列表转换为PyTorch张量
#word_indexes_tensor = torch.LongTensor(word_indexes)

# 将索引列表输入嵌入层以获取嵌入向量
#word_embeddings = embedding_layer(word_indexes_tensor)

# 输出嵌入向量
#print(word_embeddings)

