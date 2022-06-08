import torch
from torch import nn

class ProsodicEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_lstm_layers):
        super().__init__()
        self.num_lstm_layers = num_lstm_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.prenet = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            num_layers=num_lstm_layers,
            bidirectional=True,
        )
        self.h0 = nn.Parameter(torch.rand(num_lstm_layers*2, 1, hidden_channels))
        self.c0 = nn.Parameter(torch.rand(num_lstm_layers*2, 1, hidden_channels))
        self.linear = nn.Linear(hidden_channels*2, out_channels*2)

    def encode(self, x):
        batch_size = x.size(0)
        x = self.prenet(x)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        h0 = self.h0.expand(self.num_lstm_layers*2, batch_size, self.hidden_channels).contiguous()
        c0 = self.c0.expand(self.num_lstm_layers*2, batch_size, self.hidden_channels).contiguous()
        x, (h, c) = self.rnn(x, (h0, c0))
        x = torch.cat((h[0, :, :], h[-1, :, :]), dim=1)
        x = self.linear(x)
        mu, log_var = torch.split(x, self.out_channels, dim=1)
        return (mu, log_var)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        kl_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kl_term
