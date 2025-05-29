import torch
import torch.nn as nn
import torch.optim as optim
from multiDim.FunctionND import FunctionND
import math
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt

from package1.Approximator import Approximator


#-------Positional Encoding-----------
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000, encoding_type='sinusoidal'):
        super().__init__()
        self.encoding_type = encoding_type.lower()
        self.dim_model = dim_model
        if self.encoding_type == 'sinusoidal':
            pe = torch.zeros(max_len, dim_model)
            position = torch.arange(start = 0, end = max_len,step = 1, dtype=torch.float).unsqueeze(1) # Returns a 1-D tensor of size "max length" from 0 to max length-1 in accending steps of 1 [0,1,2,....,max lenght-1]
            div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)) # for each x in [0,1,2,....,dim_model-1] do x = x * -log(10000) / dim_model
            pe[:, 0::2] = torch.sin(position * div_term) # for each even position (x) starting at 0: pe = sin(x * div_term)
            pe[:, 1::2] = torch.cos(position * div_term) # for each odd position (x) starting at 1: pe = cos(x * div_term)
            self.register_buffer('pe', pe)

        elif self.encoding_type == 'learned':
            self.pe = nn.Embedding(max_len, dim_model)

        elif self.encoding_type == 'rope':
            assert dim_model % 2 == 0, "RoPE requires even model dimension"
            self.pe = None  # no explicit buffer needed

        elif self.encoding_type == 'none':
            self.pe = None

        else:
            raise ValueError(f"Unsupported encoding_type: {self.encoding_type}")

    def forward(self, x):
        # x: (batch, seq_len, dim_model)
        if self.encoding_type == 'sinusoidal':
                x = x + self.pe[:x.size(1)]

        elif self.encoding_type == 'learned':
            positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pe(positions)

        elif self.encoding_type == 'rope':
            # Rotary Positional Encoding (RoPE)
            batch, seq_len, dim = x.shape
            half_dim = dim // 2
            position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # (seq_len, 1)
            dim_range = torch.arange(0, half_dim, device=x.device).unsqueeze(0)  # (1, dim//2)
            inv_freq = 1.0 / (10000 ** (dim_range / half_dim))
            angle = position * inv_freq  # (seq_len, dim//2)
            cos = torch.cos(angle).unsqueeze(0)  # (1, seq_len, dim//2)
            sin = torch.sin(angle).unsqueeze(0)

            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x = torch.cat([
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ], dim=-1)

        # else: encoding_type == 'none' → do nothing
        return x
#-------FeedForwardNN-----------
class FeedForwardNN(nn.Module):
    def __init__(self, dim_model, hidden_dims, activation_fn='relu'):
        super().__init__()

        # Choose activation function
        if activation_fn == 'relu':
            activation = nn.ReLU()
        elif activation_fn == 'gelu':
            activation = nn.GELU()
        elif activation_fn == 'tanh':
            activation = nn.Tanh()
        elif activation_fn == 'sigmoid':
            activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        layers = []
        input_dim = dim_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, dim_model))  # Project back to original dimension

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#-------one Encoder layer-----------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, hidden_dims, activation_fn='relu', dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout)
        self.ffnn = FeedForwardNN(dim_model, hidden_dims, activation_fn)

        self.ln1 = nn.LayerNorm(dim_model)
        self.ln2 = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)

        ffnn_output = self.ffnn(x.transpose(0, 1)).transpose(0, 1)
        x = x + self.dropout(ffnn_output)
        x = self.ln2(x)

        return x

#-------Transformer usable-----------
class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, num_heads, hidden_dims, num_layers, activation_fn='relu', dropout=0.1,max_len=5000):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, hidden_dims, activation_fn, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

#-------Transformer put together (adds out and in + positional encoding)-----------
class CreateTransformer(nn.Module):
    def __init__(
        self,
        dim_model=64,
        num_heads=4,
        hidden_dims=None,
        num_layers=4,
        max_len=23000,#change this later
        pos_encoding_type='sinusoidal',
        dropout=0.1,
        activation_fn='relu',
        inputdimensions = 1,
        outputdimensions = 1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128]
        self.input_proj = nn.Linear(inputdimensions, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model, max_len, pos_encoding_type)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, hidden_dims, activation_fn, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(dim_model, outputdimensions)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.transpose(0, 1)
        x = self.output_proj(x)
        return x
####



####
# --------- transformer for use  (das usen Timo) ---------
class Approximator_Transformer(Approximator):
    def __init__(
        self,
        name="NaN",
        dim_model=64, #----Amount of dimensions a vector embedding has
        num_heads=4,  #----Amount of heads in multi head attention
        num_layers=4, #-----Amount of encoder Layers
        max_len=200, #-----Amount point to be handheld
        pos_encoding_type='sinusoidal', #----type of positional encoding
        dropout=0.1, #----dropout value
        nnactivation_function= 'relu',
        inputdimensions = 2,
        outputdimensions = 1,
        params=None,

    ):

        if params is None:
            params = [1200, 50, [8, 8, 8, 8]]
        self.epochSum = 0
        self.function = 0
        self.name=name
        self.dim_model=dim_model #----Amount of dimensions a vector embedding has
        self.num_heads=num_heads  #----Amount of heads in multi head attention
        self.num_layers=num_layers #-----Amount of encoder Layers
        self.max_len=max_len #-----Amount point to be handheld
        self.pos_encoding_type=pos_encoding_type #----type of positional encoding
        self.dropout=dropout #----dropout value
        self.nnactivation_function= nnactivation_function
        

        self.epochs = params[0]
        self.samplePoints = params[1] #----Amount of points to be used  to
        self.nodesPerLayer = params[2]
        self.inputdimensions = inputdimensions
        self.outputdimensions = outputdimensions
        self.transformer = None
        self.criterion= None
        self.optimizer= None


    def train(self, function:FunctionND):
        
        self.function = function  # Save reference to the function
        self.inputdimensions = function.inDomainEnd
        self.outputdimensions = function.inDomainStart

        self.transformer=CreateTransformer(
            dim_model=self.dim_model,
            num_heads=self.num_heads,
            hidden_dims=self.nodesPerLayer,
            num_layers=self.num_layers,
            max_len=self.max_len,
            pos_encoding_type=self.pos_encoding_type,
            dropout=self.dropout,
            activation_fn=self.nnactivation_function,
            inputdimensions=self.inputdimensions,
            outputdimensions=self.outputdimensions,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=0.01)

        # Generate training input
        x_vals = torch.linspace(self.function.xdomainstart, self.function.xdomainend, self.samplePoints).unsqueeze(1)  # (samplePoints, 1)
        if self.inputdimensions == 2:
            x = torch.linspace(self.function.xdomainstart, self.function.xdomainend, self.samplePoints).unsqueeze(1)  # (samplePoints, 1)
            y = torch.linspace(self.function.ydomainstart, self.function.ydomainend, self.samplePoints)
            X, Y = torch.meshgrid(x, y, indexing='ij')  # (samplePoints, samplePoints)
            x_vals = torch.stack([X.ravel(), Y.ravel()], dim=1)  #(samplePoints², 2)
        if self.inputdimensions > 2:
            x_vals = x_vals.repeat(1, self.inputdimensions)  #(samplePoints, inputdimensions)

        # Apply the target function
        y_vals = torch.stack([function.evaluate(x[0],x[1]) for x in x_vals], dim=0)  # (samplePoints, outputdimensions)

        # Prepare for transformer: (batch, seq_len, input/output_dim)
        x_input = x_vals.unsqueeze(0)  # (1, seq_len, input_dim)
        y_target = y_vals.unsqueeze(0)  # (1, seq_len, output_dim)

        for epoch in range(self.epochs):
            self.transformer.train()
            self.optimizer.zero_grad()

            output = self.transformer(x_input)
            loss = self.criterion(output, y_target)

            loss.backward()
            self.optimizer.step()

            if epoch % 2 == 0 or epoch == self.epochs - 1:
                print(f"[{self.__class__.__name__}] Epoch {epoch}: Loss = {loss.item():.6f}")

        self.epochSum += self.epochs

    def predict(self, X):
        self.transformer.eval()

        # Convert to tensor of shape (1, seq_len, input_dim)
        x_input = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)

        with torch.no_grad():
            prediction = self.transformer(x_input)  # (1, seq_len, output_dim)

        return prediction.squeeze(0).cpu().numpy()  # (seq_len, output_dim)


# --------- Math Data Generator ---------
class DataCreation ():
    @staticmethod
    def create1Dto1Dmathdata(negativerange : float = -math.pi, positiverange : float = math.pi, steps : int = 200, math_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.sin(x) + torch.cos(x)):
        x_vals = torch.linspace(negativerange, positiverange, steps).unsqueeze(1)  # [200,1]
        y_vals = math_function(x_vals)
        return x_vals, y_vals


    # Define a 2D -> 1D function (das ist eine Besipielfunktion)
    def target_function(x):
        # x is a 1D tensor of shape (2,)
        return torch.tensor([torch.sin(x[0]) + torch.cos(x[1])])  # → shape: [1]

    