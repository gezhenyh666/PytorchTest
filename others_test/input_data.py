import torch
import torch.nn as nn

sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
source_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
source_vocab_size = len(source_vocab)

target_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(target_vocab)}
target_vocab_size = len(target_vocab)
source_len = 5 # max length of input sequence
target_len = 6

def make_data(sentences):
  encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
  for i in range(len(sentences)):
    encoder_input = [source_vocab[word] for word in sentences[i][0].split()]
    decoder_input = [target_vocab[word] for word in sentences[i][1].split()]
    decoder_output = [target_vocab[word] for word in sentences[i][2].split()]
    encoder_inputs.append(encoder_input)
    decoder_inputs.append(decoder_input)
    decoder_outputs.append(decoder_output)
  print(encoder_input)
  print(torch.LongTensor(encoder_inputs))
  vocab_size = 1000
  embedding_dim = 100
  embedding_layer = nn.Embedding(vocab_size, embedding_dim)
  word_embeddings = embedding_layer(torch.LongTensor(encoder_inputs))
  print(word_embeddings.shape)
  return torch.LongTensor(encoder_inputs), torch.LongTensor(decoder_inputs), torch.LongTensor(decoder_outputs)

if __name__=="__main__":
  make_data(sentences)