        
import os
import random
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
__PATH__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(__PATH__)  

BATCH_SIZE = 64
MAX_SEQ_LEN = 1024
INPUT_DIM = 32
HIDDEM_DIM = 16
def prepare_pack_padded_sequence( inputs_words, seq_lengths, descending=True):
    """
    for rnn model
    :param device:
    :param inputs_words:
    :param seq_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=descending)
    _, desorted_indices = torch.sort(indices, descending=False)
    sorted_inputs_words = inputs_words[indices]
    return sorted_inputs_words, sorted_seq_lengths, desorted_indices

lstm = nn.LSTM(INPUT_DIM, HIDDEM_DIM, num_layers=2, batch_first=True, bidirectional=False)

x = torch.ones((BATCH_SIZE, MAX_SEQ_LEN, INPUT_DIM))

print(x.shape)
lengths = random.choices(range(10, 1024), k=64)
lengths.sort()
packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        
        # 通过LSTM hidden: num_layers * num_directions, batch, hidden_size
packed_output, (hidden, cell) = lstm(packed_input)
output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch_size,seq_len,hidden_dim * num_directionns ]
batch_size, max_seq_len,hidden_dim = output.shape
hidden = torch.mean(torch.reshape(hidden,[batch_size,-1,hidden_dim]),dim=1)
output = torch.sum(output,dim=1)
fc_input = output + hidden
print(fc_input.size())
        
