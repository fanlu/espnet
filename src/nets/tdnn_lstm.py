import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import PackedSequence

# SqueezeExitation Net for 1dim features
class SquExiNet(nn.Module):
  def __init__(self, input_dim, s_ratio):
    super(SquExiNet, self).__init__()
    model_list = OrderedDict()
    model_list["proj1"] = nn.Linear(input_dim, int(input_dim / s_ratio))
    model_list["relu"] = nn.ReLU()
    model_list["proj2"] = nn.Linear(int(input_dim / s_ratio), input_dim)
    model_list["relu"] = nn.ReLU()
    self.model = nn.Sequential(model_list)

  def forward(self, input):
    # input: batch_size, dim, seqLength
    input_z = input.mean(2)
    s = self.model(input_z)
    length = input.size(2)
    output = input * s.unsqueeze(-1).expand(input.size())
    return output
# todo 1 ReLU v.s. SeLU 2. batchnorm before or after SqueezeExitation Net?

# follow xvector style that one sublayer =  TDNN layer + ReLU + batchnorm
class TDNNStack(nn.Module):
  def __init__(self, input_dim, nndef, dropout, use_SE=False, SE_ratio=4, use_selu=False):
    super(TDNNStack, self).__init__()
    self.input_dim = input_dim
    model_list = OrderedDict()
    ly = 0
    for item in nndef.split("."):
      out_dim, k_size, dilation = [int(x) for x in item.split("_")]
      model_list["TDNN%d" % ly] = nn.Conv1d(input_dim, out_dim, k_size, dilation=dilation)
      if use_selu:
        model_list["SeLU%d" % ly] = nn.SELU()
      else:
        model_list["ReLU%d" % ly] = nn.ReLU()
        model_list["batch_norm%d" % ly] = nn.BatchNorm1d(out_dim)
      if use_SE:
        model_list["SEnet%d" % ly] = SquExiNet(out_dim, SE_ratio)
      if dropout != 0.0:
        model_list['dropout%d' % ly] = nn.Dropout(dropout)
      input_dim = out_dim
      ly = ly + 1
    self.model = nn.Sequential(model_list)
    self.output_dim = input_dim

  def forward(self, input):
    # # input: seqLength X batchSize X dim
    # # output: seqLength X batchSize X dim (similar to lstm)
    # input = input.contiguous().transpose(0, 1).transpose(1, 2)  # batchSize, dim, seqLength
    # output = self.model(input)
    # output = output.contiguous().transpose(1, 2).transpose(0, 1)  # seqLength, batchSize, dim
    # return output

    # input: B * T * D
    input = input.contiguous().transpose(1, 2) # B * D * T
    output = self.model(input)
    output = output.contiguous().transpose(1, 2) # B * T * D
    return output


class LSTMNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, rnn_layers, dropout=0.0, bidirectional=False, out_type='all'):
    super(LSTMNet, self).__init__()
    self.rnn = nn.LSTM(input_dim, hidden_dim, rnn_layers,
                       dropout=dropout, bidirectional=bidirectional)
    assert(out_type in ("all", "ALL", "END", "end", "AVE", "ave"))
    self.out_type = 1
    if out_type in ("END", "end"):
      self.out_type = 2
    elif out_type in ("ave", "AVE"):
      self.out_type = 3

    self.bidirectional = bidirectional

  # input: 3d tensor (seq_len, batch_size, fea_dim)
  #       It could be a PackedSequence to included seq_len for each seqence
  def forward(self, input):
    output, hc = self.rnn(input)
    # default ouput is ALL
    if self.out_type == 2:  # END
      if self.bidirectional:
        output = torch.cat((hc[0][-1], hc[0][-2]), 1)  # batch_size X 2*fea_dim
      else:
        output = hc[0][-1]
    elif self.out_type == 3:  # AVE
      if isinstance(output, PackedSequence):
        output = output[0]
      output = output.mean(0)
    return output

class TDNNLSTM(nn.Module):
  def __init__(self, idim, dropout):
    super(TDNNLSTM, self).__init__()
    from tdnn_lstm import TDNNStack
    tdnndef1 = "512_5_1.512_3_1.512_3_1"
    
    # D, def, dropout
    self.tdnn1 = TDNNStack(idim, tdnndef1, dropout)
    output_dim = self.tdnn1.output_dim
    hidden_lstm_dim = 256
    self.rnn = nn.LSTM(output_dim, hidden_lstm_dim, 1,
                        dropout=dropout, bidirectional=False, batch_first=True)
    tdnndef2 = "512_3_3.512_3_3"
    self.tdnn2 = TDNNStack(hidden_lstm_dim, tdnndef2, dropout)
    output_dim_tdnn2 = self.tdnn2.output_dim
    self.rnn2 = nn.LSTM(output_dim_tdnn2, hidden_lstm_dim, 1,
                        dropout=dropout, bidirectional=False, batch_first=True)
    

  def forward(self, xs_pad, ilens):
    '''BLSTM forward

    :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
    :param torch.Tensor ilens: batch of lengths of input sequences (B)
    :return: batch of hidden state sequences (B, Tmax, erojs)
    :rtype: torch.Tensor
    '''
    logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
    output1 = self.tdnn1(xs_pad)
    print(output1.shape)
    output2, hc = self.rnn(output1)
    print(output2.shape)
    output3 = self.tdnn2(output2)
    print(output3.shape)
    output4, hc2 = self.rnn2(output3)
    print(output4.shape)
    return output4

if __name__ == "__main__":
  # from e2e_asr_th import TDNNLSTM
  # B T D
  input1 = torch.randn(5, 200, 3)
  tl = TDNNLSTM(3, 0.5)
  o2 = tl(input1, 2)
  print(o2.shape)
  # import os
  # print(os.environ)
  # bilstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True)
  # B,D,T
  # input1 = torch.randn(5, 3, 200)
  # h0 = torch.randn(4, 3, 20)
  # c0 = torch.randn(4, 3, 20)
  # output, (hn, cn) = bilstm(input, (h0, c0))
  # print('output shape: ', output.shape)
  # print('hn shape: ', hn.shape)
  # print('cn shape: ', cn.shape)
  tdnndef1 = "512_5_1.512_3_1.512_3_1"
  dropout = 0.5

  # T B D
  input1 = torch.randn(200, 5, 3)
  # D, def, dropout
  tdnn1 = TDNNStack(3, tdnndef1, dropout)
  # model_list = OrderedDict()
  # input_dim = 3
  # for ly, item in enumerate(items.split(".")):
  #   out_dim, k_size, dilation = [int(x) for x in item.split("_")]
  #   model_list["TDNN%d" % ly] = nn.Conv1d(input_dim, out_dim, k_size, dilation=dilation)
  #   model_list["ReLU%d" % ly] = nn.ReLU()
  #   model_list["batch_norm%d" % ly] = nn.BatchNorm1d(out_dim)
  #   input_dim = out_dim
  # model = nn.Sequential(model_list)
  output1 = tdnn1(input1)
  print(output1.shape)
  output_dim = tdnn1.output_dim
  hidden_lstm_dim = 256

  rnn = nn.LSTM(output_dim, hidden_lstm_dim, 1,
                       dropout=dropout, bidirectional=False)
  output2, hc = rnn(output1)
  print(output2.shape)

  tdnndef2 = "512_3_3.512_3_3"
  tdnn2 = TDNNStack(hidden_lstm_dim, tdnndef2, dropout)
  output3 = tdnn2(output2)
  output_dim_tdnn2 = tdnn2.output_dim
  print(output3.shape)

  rnn2 = nn.LSTM(output_dim_tdnn2, hidden_lstm_dim, 1,
                       dropout=dropout, bidirectional=False)
  output4, hc2 = rnn2(output3)
  print(output4.shape)
  
