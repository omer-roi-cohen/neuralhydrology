from torch import nn
import torch


class IntegLSTM(nn.Module):
    def __init__(self, cfg):
        super(IntegLSTM, self).__init__()
        self.cfg = cfg
        self.dynamic_input_size = len(self.cfg.dynamic_inputs)
        self.static_input_size = len(self.cfg.static_attributes)
        self.lstm_input_size = self.dynamic_input_size + self.cfg.statics_embedding['hiddens'][0]
        self.fc_input = nn.Linear(in_features=self.static_input_size, out_features=self.cfg.statics_embedding['hiddens'][0])
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.cfg.hidden_size, bias=True)
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)
        self.fc_output = nn.Linear(in_features=self.cfg.hidden_size, out_features=1)

    def forward(self, x):
        x_s = x[:, :, :self.static_input_size]
        x_d = x[:, :, self.static_input_size:]
        fc_i_out = self.fc_input.forward(x_s)
        lstm_input = torch.cat((fc_i_out, x_d),2)
        output, (h_n, c_n) = self.lstm.forward(lstm_input)
        pred = self.fc_output(self.dropout(h_n[-1, :, :]))
        return pred

    def copy_weights(self, cuda_lstm):
        fc_input_state_dict = self.fc_input.state_dict()
        fc_input_state_dict['weight'].copy_(cuda_lstm.embedding_net.state_dict()['statics_embedding.net.0.weight'].data)
        fc_input_state_dict['bias'].copy_(cuda_lstm.embedding_net.state_dict()['statics_embedding.net.0.bias'].data)
        self.lstm.load_state_dict(cuda_lstm.lstm.state_dict())
        fc_output_state_dict = self.fc_output.state_dict()
        fc_output_state_dict['weight'].copy_(cuda_lstm.head.state_dict()['net.0.weight'].data)
        fc_output_state_dict['bias'].copy_(cuda_lstm.head.state_dict()['net.0.bias'].data)
