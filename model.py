import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_e2e_autoencoder(cfg):

    # initialize encoder and decoder
    encoder = E2E_Encoder(in_channels=cfg['in_channels'],
                          n_electrodes=cfg['n_electrodes'],
                          out_scaling=cfg['output_scaling'],
                          out_activation=cfg['encoder_out_activation']).to(cfg['device'])

    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          out_activation=cfg['decoder_out_activation']).to(cfg['device'])
    
    # If output steps are specified, add safety layer at the end of the encoder model 
    if cfg['output_steps'] != 'None':
        assert cfg['encoder_out_activation'] == 'sigmoid'
        encoder.output_scaling = 1.0
        encoder = torch.nn.Sequential(encoder,
                                      SafetyLayer(n_steps=10,
                                                  order=2,
                                                  out_scaling=cfg['output_scaling'])).to(cfg['device'])
    return encoder, decoder

def get_Zhao_autoencoder(cfg):
    encoder = ZhaoEncoder(in_channels=cfg['in_channels'], n_electrodes=cfg['n_electrodes']).to(cfg['device'])
    decoder = ZhaoDecoder(out_channels=cfg['out_channels'], out_activation=cfg['decoder_out_activation']).to(cfg['device'])

    return encoder, decoder

def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


def convlayer3d(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer 

def deconvlayer3d(n_input, n_output, k_size=2, stride=2, padding=0, dilation=1, resample_out=None):
    layer = [
        nn.ConvTranspose3d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(n_output),
        nn.LeakyReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.resample_out = resample_out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        if self.resample_out:
            out = self.resample_out(out)
        return out
    
  
class SafetyLayer(torch.nn.Module):
    def __init__(self, n_steps=5, order=1, out_scaling=120e-6):
        super(SafetyLayer, self).__init__()
        self.n_steps = n_steps
        self.order = order
        self.output_scaling = out_scaling

    def stairs(self, x):
        """Assumes input x in range [0,1]. Returns quantized output over range [0,1] with n quantization levels"""
        return torch.round((self.n_steps-1)*x)/(self.n_steps-1)

    def softstairs(self, x):
        """Assumes input x in range [0,1]. Returns sin(x) + x (soft staircase), scaled to range [0,1].
        param n: number of phases (soft quantization levels)
        param order: number of recursion levels (determining the steepnes of the soft quantization)"""

        return (torch.sin(((self.n_steps - 1) * x - 0.5) * 2 * math.pi) +
                         (self.n_steps - 1) * x * 2 * math.pi) / ((self.n_steps - 1) * 2 * math.pi)
    
    def forward(self, x):
        out = self.softstairs(x) + self.stairs(x).detach() - self.softstairs(x).detach()
        return (out * self.output_scaling).clamp(1e-32,None)


class VGGFeatureExtractor():
    def __init__(self,layer_names=['1','3','6','8'], layer_depth=9 ,device='cpu'):
        
        # Load the VGG16 model
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*[*model.features][:layer_depth]).to(device)
        
        # Register a forward hook for each layer of interest
        self.layers = {name: layer for name, layer in self.feature_extractor.named_children() if name in layer_names}
        self.outputs = dict()
        for name, layer in self.layers.items():
            layer.__name__ = name
            layer.register_forward_hook(self.store_output)
            
    def store_output(self, layer, input, output):
        self.outputs[layer.__name__] = output

    def __call__(self, x):
        
        # If grayscale, convert to RGB
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        
        # Forward pass
        self.feature_extractor(x)
        activations = list(self.outputs.values())
        
        return activations
        
        

class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """
    def __init__(self, in_channels=3, out_channels=1, n_electrodes=638, out_scaling=1e-4, out_activation='relu'):
        super(E2E_Encoder, self).__init__()
        self.output_scaling = out_scaling
        self.out_activation = {'tanh': nn.Tanh(), ## NOTE: simulator expects only positive stimulation values 
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.ReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,8,3,1,1),
                                   *convlayer(8,16,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   *convlayer(16,32,3,1,1,resample_out=nn.MaxPool2d(2)),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   ResidualBlock(32, resample_out=None),
                                   *convlayer(32,16,3,1,1),
                                   nn.Conv2d(16,1,3,1,1),
                                   nn.Flatten(),
                                   nn.Linear(1024,n_electrodes),
                                   self.out_activation)

    def forward(self, x):
        self.out = self.model(x)
        stimulation = self.out*self.output_scaling #scaling improves numerical stability
        print(f"Stimulation range: [{stimulation.min():.3f}, {stimulation.max():.3f}]")
        return stimulation

class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()

        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1),
                                   self.out_activation)

    def forward(self, x):
        return self.model(x)

# class E2E_RealisticPhospheneSimulator(nn.Module):
#     """A realistic simulator, using stimulation vectors to form a phosphene representation
#     in: a 1024 length stimulation vector
#     out: 256x256 phosphene representation
#     """
#     def __init__(self, cfg, params, r, phi):
#         super(E2E_RealisticPhospheneSimulator, self).__init__()
#         self.simulator = GaussianSimulator(params, r, phi, batch_size=cfg.batch_size, device=cfg.device)
        
#     def forward(self, stimulation):
#         phosphenes = self.simulator(stim_amp=stimulation).clamp(0,1)
#         phosphenes = phosphenes.view(phosphenes.shape[0], 1, phosphenes.shape[1], phosphenes.shape[2])
#         return phosphenes

class ZhaoEncoder(nn.Module):
    def __init__(self, in_channels=3,n_electrodes=638, out_channels=1):
        super(ZhaoEncoder, self).__init__()

        self.model = nn.Sequential(
            *convlayer3d(in_channels,32,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(32,48,3,1,1, resample_out=nn.MaxPool3d(2,(1,2,2),padding=(1,0,0),dilation=(2,1,1))),
            *convlayer3d(48,64,3,1,1),
            *convlayer3d(64,1,3,1,1),

            nn.Flatten(start_dim=3),
            nn.Linear(1024,n_electrodes),
            nn.ReLU()
        )

    def forward(self, x):
        self.out = self.model(x)
        self.out = self.out.squeeze(dim=1)
        self.out = self.out*1e-4
        return self.out

class ZhaoDecoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(ZhaoDecoder, self).__init__()
        
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]

        self.model = nn.Sequential(
            *convlayer3d(in_channels,16,3,1,1),
            *convlayer3d(16,32,3,1,1),
            *convlayer3d(32,64,3,(1,2,2),1),
            *convlayer3d(64,32,3,1,1),
            nn.Conv3d(32,out_channels,3,1,1),
            self.out_activation
        )

    def forward(self, x):
        self.out = self.model(x)
        return self.out
    
    
### Exp 3 interaction models (Basic coactivation models)

def ignore_inactive_electrodes(interaction_model, threshold=20e-6):
    def wrapper(stim, threshold=threshold):
        """Returns coactivation_model(x) only for active electrodes (above-threshold).
        For inactive electrodes the return value is set to x."""
        output = interaction_model(stim)
        active_electrodes = torch.ones_like(stim).masked_fill(stim<threshold,0)
        inactive_electrodes = (1-active_electrodes).clip(0,1)
        return output * active_electrodes + stim * inactive_electrodes
    return wrapper
        

# Interaction model
def get_interaction_model(electrode_coords, data_kwargs, interaction):   
    # Initialize interaction layer
    n_electrodes = len(electrode_coords)
    n_phosphenes = len(electrode_coords)
    interaction_model = torch.nn.Linear(n_electrodes, n_phosphenes, bias=True).to(data_kwargs['device'])
    interaction_model.weight.requires_grad = False
    interaction_model.bias.requires_grad = False
    
    # Squared distance between electrodes
    x, y = electrode_coords.cartesian
    dist_squared = ((x[:,None]-x)**2 + (y[:,None]-y)**2)
    
    if interaction == 'no-interaction':
        interaction_model.bias.data = torch.zeros(n_phosphenes, **data_kwargs)
        interaction_model.weight.data = torch.eye(n_phosphenes, n_electrodes,**data_kwargs)

    elif interaction == 'electr-coactivation':
        SCALE = 100 
        inter_weight = torch.from_numpy(1/(1+SCALE*dist_squared)).to(**data_kwargs)
        interaction_model.weight.data = inter_weight
        interaction_model = ignore_inactive_electrodes(interaction_model)
        
    elif interaction == 'costimulation-loss':
        neigh_weight = torch.from_numpy(1/(1+dist_squared)).to(**data_kwargs)
        diag_mask = torch.eye(n_phosphenes,n_electrodes,dtype=bool, device=data_kwargs['device'])
        neigh_weight = neigh_weight.masked_fill(diag_mask,0)
        interaction_model.weight.data = neigh_weight
    
    else:
        raise NotImplementedError(f'Undefined interaction: {interaction}')

    return interaction_model

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network that uses a pretrained EfficientNet_B0 
    for feature extraction and an RNN layer for temporal processing
    """
    def __init__(self, in_channels=3, n_electrodes=638, out_scaling=1e-4, 
                 out_activation='relu', rnn_type='lstm', hidden_size=256):
        super(CRNN, self).__init__()
        self.output_scaling = out_scaling
        self.hidden_size = hidden_size
        
        # Define output activation
        self.out_activation = {'tanh': nn.Tanh(),
                              'sigmoid': nn.Sigmoid(),
                              'relu': nn.ReLU(),
                              'softmax': nn.Softmax(dim=1)}[out_activation]
        
        # Load pretrained EfficientNet_B0
        self.feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Replace first conv layer if input channels != 3
        if in_channels != 3:
            print(f"Changing input channels from {in_channels} to 3")
            
            self.feature_extractor.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Remove the classifier head
        self.feature_extractor = self.feature_extractor.features
        
        # Get feature dimension from EfficientNet (1280 for B0)
        self.feature_dim = 1280
        
        # RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(self.feature_dim, hidden_size, batch_first=True)
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(self.feature_dim, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
            
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, n_electrodes),
            self.out_activation
        )

    def forward(self, x):
        # MNIST gives us [batch_size, channels, height, width]
        # but our model expects [batch_size, sequence_length, channels, height, width]
        
        # Check if input is 4D (missing sequence dimension)
        
        
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            print("Input is not sequential - adding sequence dimension")
        else:
            print("Input is sequential with shape", x.shape)
            batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape for CNN processing
        x_reshaped = x.view(batch_size * seq_len, *x.shape[2:])
        
        # Extract features with EfficientNet
        features = self.feature_extractor(x_reshaped)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.reshape(batch_size, seq_len, self.feature_dim)
        
        # Process with RNN
        rnn_out, _ = self.rnn(features)
        
        # Get output for last time step
        last_output = rnn_out[:, -1, :]
        
        # Generate stimulation pattern
        stimulation = self.fc(last_output)
        
        # Apply scaling
        stimulation = stimulation * self.output_scaling
        
        return stimulation

def get_CRNN_autoencoder(cfg):
    """
    Returns a CRNN encoder and compatible decoder based on configuration
    """
    # Initialize encoder
    encoder = CRNN(in_channels=cfg['in_channels'],
                   n_electrodes=cfg['n_electrodes'],
                   out_scaling=cfg['output_scaling'],
                   out_activation=cfg['encoder_out_activation'],
                   rnn_type=cfg.get('rnn_type', 'lstm'),
                   hidden_size=cfg.get('hidden_size', 256)).to(cfg['device'])
    
    # Initialize decoder (reuse existing E2E_Decoder)
    decoder = E2E_Decoder(out_channels=cfg['out_channels'],
                          out_activation=cfg['decoder_out_activation']).to(cfg['device'])
    
    # Apply safety layer if specified
    if cfg.get('output_steps', 'None') != 'None':
        assert cfg['encoder_out_activation'] == 'sigmoid'
        encoder.output_scaling = 1.0
        encoder = torch.nn.Sequential(encoder,
                                     SafetyLayer(n_steps=10,
                                                order=2,
                                                out_scaling=cfg['output_scaling'])).to(cfg['device'])
    
    return encoder, decoder

class SimpleCRNN(nn.Module):
    """Simplified CRNN for debugging"""
    def __init__(self, in_channels=1, n_electrodes=60, out_scaling=1.0, 
                 out_activation='sigmoid'):
        super(SimpleCRNN, self).__init__()
        self.output_scaling = out_scaling
        self.n_electrodes = n_electrodes
        
        self.out_activation = {'relu': nn.ReLU(),
                              'sigmoid': nn.Sigmoid()}[out_activation]
        
        # CNN backbone similar to E2E_Encoder
        self.feature_extractor = nn.Sequential(
            *convlayer(in_channels, 8, 3, 1, 1),
            *convlayer(8, 16, 3, 1, 1, resample_out=nn.MaxPool2d(2)),
            *convlayer(16, 32, 3, 1, 1, resample_out=nn.MaxPool2d(2)),
            ResidualBlock(32, resample_out=None),
            *convlayer(32, 16, 3, 1, 1),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Standard LSTM without layer_norm
        self.rnn = nn.LSTM(8, 32, batch_first=True)
        
        # Add LayerNorm *after* the LSTM
        self.layer_norm = nn.LayerNorm(32)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(32, n_electrodes),
            self.out_activation
        )
        
    def forward(self, x):
        # Handle both 4D and 5D inputs
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
            #print("Input is not sequential - adding sequence dimension")
        else:
            #print("Input is sequential with shape", x.shape)
        
            batch_size, seq_len = x.size(0), x.size(1)
        batch_size, seq_len = x.size(0), x.size(1)
        # Process each sequence element with CNN
        features = []
        for i in range(seq_len):
            feat = self.feature_extractor(x[:, i])
            features.append(feat.squeeze(-1).squeeze(-1))
            
            # Debug CNN output
            #print(f"CNN features range step {i}: [{feat.min():.3f}, {feat.max():.3f}]")
        
        features = torch.stack(features, dim=1)
        #print(f"Stacked features shape: {features.shape}")
        #print(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
        
        # RNN with gradient clipping
        rnn_out, _ = self.rnn(features)
        
        # Apply layer normalization to the output
        normalized_out = self.layer_norm(rnn_out)
        
        # Get final output
        last_output = normalized_out[:, -1]
        out = self.fc(last_output)
        
        # Apply scaling
        stimulation = out * self.output_scaling
        #print(f"Stimulation range: [{stimulation.min():.3f}, {stimulation.max():.3f}]")
        return stimulation

def get_debug_CRNN_autoencoder(cfg):
    """Returns a simplified CRNN for debugging"""
    encoder = SimpleCRNN(
        in_channels=cfg['in_channels'],
        n_electrodes=cfg['n_electrodes'],
        out_scaling=cfg['output_scaling'],
        out_activation=cfg['encoder_out_activation']
    ).to(cfg['device'])
    
    decoder = E2E_Decoder(
        out_channels=cfg['out_channels'],
        out_activation=cfg['decoder_out_activation']
    ).to(cfg['device'])
    
    return encoder, decoder
