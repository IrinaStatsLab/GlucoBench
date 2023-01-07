import torch


class MLP(torch.nn.Module):
    def __init__(self, 
                 total_time_steps: int,
                 num_features: int, 
                 num_decoder_steps: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float
                 ):
        super(MLP, self).__init__()

        self.projection = torch.nn.Linear(num_features, 1)
        self.layers = [torch.nn.Linear(total_time_steps, hidden_size), 
                       torch.nn.ReLU(), 
                       torch.nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(dropout))
        self.layers.append(torch.nn.Linear(hidden_size, num_decoder_steps))
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.projection(x).squeeze(-1)
        x = self.layers(x)
        return x