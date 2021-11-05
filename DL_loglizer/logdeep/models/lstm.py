import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        input0, input1 = features[0], features[1]

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        # self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class robustlog_s(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog_s, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(input0, (h0, c0))
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.sum(h_n, dim=1)
        h_n = h_n.squeeze(dim=1)
        attention_w = self.attention_weights_layer(h_n)
        attention_w = attention_w.unsqueeze(dim=1)
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))
        softmax_w = torch.nn.functional.softmax(attention_context, dim=-1)
        out = torch.bmm(softmax_w, out)

        out = self.fc(out[:, -1, :])
        return out


class robustlog_bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog_bi, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            # batch_first=True,
                            bidirectional=True
                            )
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(hidden_size, num_keys)
        # self.fc = nn.Linear(2 * hidden_size, num_keys)

    def forward(self, features, device):
        # print('=========>',features[0].dtype)
        # print(features[0].shape)
        # features = torch.from_numpy(np.array(features))
        fs = features[0].permute(1, 0, 2)
        input0 = fs
        h0 = torch.zeros(self.num_layers*2, input0.size(1),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, input0.size(1),
                         self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(input0, (h0, c0))
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2) 
        out = forward_out + backward_out #[seq_len, batch, hidden_size] 
        out = out.permute(1, 0, 2) #[batch, seq_len, hidden_size]
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.sum(h_n, dim=1)
        h_n = h_n.squeeze(dim=1)
        attention_w = self.attention_weights_layer(h_n)
        attention_w = attention_w.unsqueeze(dim=1)
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))
        softmax_w = torch.nn.functional.softmax(attention_context, dim=-1)
        out = torch.bmm(softmax_w, out)

        out = self.fc(out[:, -1, :])
        return out