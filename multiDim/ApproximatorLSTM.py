from multiDim import FunctionND
from multiDim.ApproximatorND import ApproximatorND
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LSTMModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 dropout=0.0,
                 activation='relu',
                 bidirectional=False):
        super(LSTMModel, self).__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        direction_multiplier = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=bidirectional,
                            batch_first=True)

        # Optional fully connected block
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction_multiplier, hidden_size),
            self._get_activation(activation),
            nn.Linear(hidden_size, output_size)
        )

    @staticmethod
    def _get_activation(name):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU()
        }.get(name.lower(), nn.ReLU())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # x: (batch, seq_len, input_size)
        last_output = lstm_out[:, -1, :]  # Take output of last time step
        return self.fc(last_output)       # Output: (batch, output_size)


class LSTMApproximatorCreate(ApproximatorND):
    def __init__(self,
        name="NaN",
        dropout=0.1, #----dropout value
        nnactivation_function= 'relu',
        device= "cpu",
        lossfunction=nn.MSELoss,
        num_layers=1,
        hidden_size=8, #----how "mutch" the lstm can remember
        lr=0.001,
        params=None,
        batch_size=32,
        bidirectional=False
        ):
        super().__init__(name,params)
        if params is None:
            params = [500, 500]
        self.name=name
        self.dropout=dropout #----dropout value
        self.nnactivation_function= nnactivation_function
        self.device= device
        self.hidden_size=hidden_size
        self.num_layers= num_layers
        self.lossfunction=lossfunction
        self.lr=lr
        self.batch_size=batch_size
        self.bidirectional = bidirectional
        self.epochs = params[0]
        self.samplePoints = params[1] #----Amount of points to be used  to

        self.lsmt=None,
        self.criterion = None
        self.optimizer = None

    def train(self, function: FunctionND):
        self.function = function
        self.inputdimensions = function.inputDim
        self.outputdimensions = function.outputDim

        self.lsmt = LSTMModel(
            input_size=self.inputdimensions,
            hidden_size=self.hidden_size,
            output_size=self.outputdimensions,
            num_layers=self.num_layers,
            dropout=self.dropout,
            activation=self.nnactivation_function,
            bidirectional=self.bidirectional,
        ).to(self.device)

        self.criterion = self.lossfunction()
        self.optimizer = optim.Adam(self.lsmt.parameters(), lr=self.lr)

        # --- Generate synthetic training data
        X = np.random.uniform(function.inDomainStart, function.inDomainEnd, size=(self.samplePoints, function.inputDim))
        y = function.evaluate(X)
        y = function.format_output_shape(y)

        # --- Create sequences
        sequence_length = min(10, len(X))  # or set explicitly
        X_seq = []
        y_seq = []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32).to(self.device)
        y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # --- Training loop
        self.lsmt.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                preds = self.lsmt(xb)

                # Ensure target shape matches predictions
                if preds.shape != yb.shape:
                    yb = yb.view_as(preds)

                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss:.4f}")

    def predict(self, input):
        self.lsmt.eval()
        with torch.no_grad():
            x = torch.tensor(input, dtype=torch.float32).unsqueeze(1).to(self.device)  # shape: [batch, 1, input_dim]
            output = self.lsmt(x)  # output: [batch, output_dim]
            return output.cpu().numpy().reshape(-1, self.outputdimensions)

    #def predict(self, input: np.ndarray):
       # self.lsmt.eval()
       # input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dim
       # with torch.no_grad():
       #     pred = self.lsmt(input_tensor)
       # return pred.cpu().numpy()

