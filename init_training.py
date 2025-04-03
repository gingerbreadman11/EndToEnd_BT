import os
import torch
import numpy as np
import pickle

import dynaphos
from dynaphos.cortex_models import get_visual_field_coordinates_probabilistically
from dynaphos.simulator import GaussianSimulator as PhospheneSimulator
from dynaphos.utils import get_data_kwargs

import model

import local_datasets
from torch.utils.data import DataLoader
from utils import resize, normalize, undo_standardize, dilation3x3, CustomSummaryTracker

from torch.utils.tensorboard import SummaryWriter


class LossTerm():
    """Loss term that can be used for the compound loss"""

    def __init__(self, name=None, func=torch.nn.functional.mse_loss, arg_names=None, weight=1.):
        self.name = name
        self.func = func  # the loss function
        self.arg_names = arg_names  # the names of the inputs to the loss function
        self.weight = weight  # the relative weight of the loss term


class CompoundLoss():
    """Helper class for combining multiple loss terms. Initialize with list of
    LossTerm instances. Returns dict with loss terms and total loss"""

    def __init__(self, loss_terms):
        self.loss_terms = loss_terms

    def __call__(self, loss_targets):
        """Calculate all loss terms and the weighted sum"""
        self.out = dict()
        self.out['total'] = 0
        for lt in self.loss_terms:
            func_args = [loss_targets[name] for name in lt.arg_names]  # Find the loss targets by their name
            self.out[lt.name] = lt.func(*func_args)  # calculate result and add to output dict
            self.out['total'] += self.out[lt.name] * lt.weight  # add the weighted loss term to the total
        return self.out

    def items(self):
        """return dict with loss tensors as dict with Python scalars"""
        return {k: v.item() for k, v in self.out.items()}


class RunningLoss():
    """Helper class to track the running loss over multiple batches."""

    def __init__(self):
        self.dict = dict()
        self.reset()

    def reset(self):
        self._counter = 0
        for key in self.dict.keys():
            self.dict[key] = 0.

    def update(self, new_entries):
        """Add the current loss values to the running loss"""
        self._counter += 1
        for key, value in new_entries.items():
            if key in self.dict:
                self.dict[key] += value
            else:
                self.dict[key] = value

    def get(self):
        """Get the average loss values (total loss dived by the processed batch count)"""
        out = {key: (value / self._counter) for key, value in self.dict.items()}
        return out


class L1FeatureLoss(object):
    def __init__(self):
        self.feature_extractor = model.VGGFeatureExtractor(device=device)
        self.loss_fn = torch.nn.functional.l1_loss

    def __call__(self, y_pred, y_true, ):
        true_features = self.feature_extractor(y_true)
        pred_features = self.feature_extractor(y_pred)
        err = [self.loss_fn(pred, true) for pred, true in zip(pred_features, true_features)]
        return torch.mean(torch.stack(err))



def get_dataset(cfg):
    if cfg['dataset'] == 'ADE50K':
        trainset, valset = local_datasets.get_ade50k_dataset(cfg)
    elif cfg['dataset'] == 'BouncingMNIST':
        trainset, valset = local_datasets.get_bouncing_mnist_dataset(cfg)
    elif cfg['dataset'] == 'Characters':
        trainset, valset = local_datasets.get_character_dataset(cfg)
    elif cfg['dataset'] == 'MNIST':
        trainset, valset = local_datasets.get_mnist_dataset(cfg)
    elif cfg['dataset'] == 'spiking_MNIST':
        trainset, valset = local_datasets.get_spiking_mnist_dataset(cfg)
    elif cfg['dataset'] == 'NMNIST_to_MNIST':
        # Load datasets
        nmnist_data, nmnist_data_val = local_datasets.get_nmnist_dataset(cfg)
        mnist_data, mnist_data_val = local_datasets.get_mnist_dataset(cfg)

        trainset = local_datasets.MatchingNMNISTMNISTDataset(nmnist_data, mnist_data, cfg['matches_train_file'])
        valset = local_datasets.MatchingNMNISTMNISTDataset(nmnist_data_val, mnist_data_val, cfg['matches_test_file'])

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, drop_last=True)
    example_batch = next(iter(valloader))
    
    # ------------------------------------------------
    size = 128  
    mask = torch.ones(1, size, size)
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    mask *= (x.pow(2) + y.pow(2)) <= 1
    
    cfg['circular_mask'] = mask.to(cfg['device'])
    # ------------------------------------------------ fix: quick fix for MNIST (maybe make sure if it makes sense)
    dataset = {'trainset': trainset,
               'valset': valset,
               'trainloader': trainloader,
               'valloader': valloader,
               'example_batch': example_batch}

    return dataset


def get_models(cfg):
    if cfg['model_architecture'] == 'end-to-end-autoencoder':
        encoder, decoder = model.get_e2e_autoencoder(cfg)
    elif cfg['model_architecture'] == 'zhao-autoencoder':
        encoder, decoder = model.get_Zhao_autoencoder(cfg)
    elif cfg['model_architecture'] == 'crnn-autoencoder':
        encoder, decoder = model.get_CRNN_autoencoder(cfg)
    elif cfg['model_architecture'] == 'debug-crnn':
        encoder, decoder = model.get_debug_CRNN_autoencoder(cfg)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=cfg['learning_rate'])

    simulator = get_simulator(cfg)

    models = {'encoder' : encoder,
              'decoder' : decoder,
              'optimizer': optimizer,
              'simulator': simulator,}
    
    # added section for exp3 with interaction layer
    if 'interaction' in cfg.keys(): 
        with open(cfg['electrode_coords'], 'rb') as handle:
            electrode_coords = pickle.load(handle)
        models['interaction'] = model.get_interaction_model(electrode_coords, simulator.data_kwargs, cfg['interaction'])

    return models


def get_simulator(cfg):
    # initialise simulator
    params = dynaphos.utils.load_params(cfg['base_config'])
    params['run'].update(cfg)
    params['thresholding'].update(cfg)
    device = get_data_kwargs(params)['device']

    with open(cfg['phosphene_map'], 'rb') as handle:
        coordinates_visual_field = pickle.load(handle, )
    simulator = PhospheneSimulator(params, coordinates_visual_field)
    cfg['SPVsize'] = simulator.phosphene_maps.shape[-2:]
    return simulator


def get_logging(cfg):
    out = dict()
    out['training_loss'] = RunningLoss()
    out['validation_loss'] = RunningLoss()
    out['tensorboard_writer'] = SummaryWriter(os.path.join(cfg['save_path'], 'tensorboard/'))
    out['training_summary'] = CustomSummaryTracker()
    out['validation_summary'] = CustomSummaryTracker()
    out['example_output'] = CustomSummaryTracker()
    return out

####### ADJUST OR ADD TRAINING PIPELINE BELOW

def get_training_pipeline(cfg):
    if cfg['pipeline'] == 'unconstrained-image-autoencoder':
        forward, lossfunc = get_pipeline_unconstrained_image_autoencoder(cfg)
    elif cfg['pipeline'] == 'constrained-image-autoencoder':
        forward, lossfunc = get_pipeline_constrained_image_autoencoder(cfg)
    elif cfg['pipeline'] == 'supervised-boundary-reconstruction':
        forward, lossfunc = get_pipeline_supervised_boundary_reconstruction(cfg)
    elif cfg['pipeline'] == 'unconstrained-video-reconstruction':
        forward, lossfunc = get_pipeline_unconstrained_video_reconstruction(cfg)
    elif cfg['pipeline'] == 'image-autoencoder-interaction-model':
        forward, lossfunc = get_pipeline_interaction_model(cfg)
    elif cfg['pipeline'] == 'image-autoencoder-coactivation-loss':
        forward, lossfunc = get_pipeline_coactivation_loss(cfg)
    else:
        print(cfg['pipeline'] + 'not supported yet')
        raise NotImplementedError

    return {'forward': forward, 'compound_loss_func': lossfunc}

def get_pipeline_unconstrained_image_autoencoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""
        #print("Starting forward pass...")
        
        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        #print(f"Input image shape: {image.shape}")
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        #print("Resetting simulator...")
        simulator.reset()
        #print("Running encoder...")
        stimulation = encoder(image)
        
        # Add debug print to check stimulation values
        stim_min = torch.min(stimulation).item()
        stim_max = torch.max(stimulation).item()
        stim_mean = torch.mean(stimulation).item()
        print(f"Stimulation range: min={stim_min:.8f}, max={stim_max:.8f}, mean={stim_mean:.8f}")
        
        #print("Running simulator...")
        phosphenes = simulator(stimulation).unsqueeze(1)
        #print(f"Phosphenes shape: {phosphenes.shape}")
        #print("Running decoder...")
        reconstruction = decoder(phosphenes)
        #print(f"Reconstruction shape: {reconstruction.shape}")

        # Output dictionary
        out = {'input':  unstandardized_image * cfg['circular_mask'],
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}
        
        #print("Forward pass complete!")
        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_constrained_image_autoencoder(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        unstandardized_image = undo_standardize(image) # image values scaled back to range 0-1

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary
        out = {'input':  unstandardized_image * cfg['circular_mask'],
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'input_resized': resize(unstandardized_image * cfg['circular_mask'], cfg['SPVsize'])}
        
        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes),
                    'input_centers': simulator.sample_centers(out['input_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'input_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_supervised_boundary_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, label = batch
        label = dilation3x3(label)

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes) * cfg['circular_mask']

        # Output dictionary
        out = {'input': image,
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'target': label * cfg['circular_mask'],
               'target_resized': resize(label * cfg['circular_mask'], cfg['SPVsize'],),}

        # Sample phosphenes and target at the centers of the phosphenes
        out.update({'phosphene_centers': simulator.sample_centers(phosphenes) ,
                    'target_centers': simulator.sample_centers(out['target_resized']) })

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'target'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_centers', 'target_centers'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_unconstrained_video_reconstruction(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        # Unpack
        frames = batch
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Forward
        simulator.reset()
        stimulation_sequence = encoder(frames).permute(1, 0, 2)  # permute: (Batch,Time,Num_phos) -> (Time,Batch,Num_phos)
        phosphenes = []
        for stim in stimulation_sequence:
            phosphenes.append(simulator(stim))  # simulator expects (Batch, Num_phosphenes)
        phosphenes = torch.stack(phosphenes, dim=1).unsqueeze(dim=1)  # Shape: (Batch, Channels=1, Time, Height, Width)
        reconstruction = decoder(phosphenes)

        out =  {'stimulation': stimulation_sequence,
                'phosphenes': phosphenes,
                'reconstruction': reconstruction * cfg['circular_mask'],
                'input': frames * cfg['circular_mask'],
                'input_resized': resize(frames * cfg['circular_mask'],
                                         (cfg['sequence_length'],*cfg['SPVsize']),interpolation='trilinear'),}

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}

        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1-cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphenes', 'input_resized'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func


def get_pipeline_interaction_model(cfg):
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        interaction_model = models['interaction']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch
        
        # Check if we have sequence data (5D tensor)
        has_sequence = len(image.shape) == 5
        
        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        interaction = interaction_model(stimulation).clip(min=0)
        phosphenes = simulator(interaction).unsqueeze(1)
        reconstruction = decoder(phosphenes)

        # Output dictionary with proper handling for both sequence and non-sequence data
        if has_sequence:
            # warning: this is not correct, we should use the whole sequence for the circular mask and resizing
            # we lose information about the sequence
            # For sequence data, we need to match dimensions
            # Original image: [batch, seq, channel, height, width]
            # Reconstruction: [batch, channel, height, width]
            
            # Take just the last frame from the sequence for comparison
            last_frame = image[:, -1]  # Shape: [batch, channel, H, W]
            
            out = {
                # Keep full sequence for reference
                'full_input': image,
                # For loss calculation use just the matching frame
                'input': last_frame * cfg['circular_mask'],
                'stimulation': stimulation,
                'interaction': interaction,
                'phosphenes': phosphenes,
                'reconstruction': reconstruction * cfg['circular_mask'],
                'input_resized': resize(last_frame * cfg['circular_mask'], cfg['SPVsize'])
            }
        else:
            # Original code for non-sequence data
            out = {'input': image * cfg['circular_mask'],
                  'stimulation': stimulation,
                  'interaction': interaction,
                  'phosphenes': phosphenes,
                  'reconstruction': reconstruction * cfg['circular_mask'],
                  'input_resized': resize(image * cfg['circular_mask'], cfg['SPVsize'])}
        
        # Target phosphene brightness is sampled pixels at centers of the phosphenes
        target_pixels = simulator.sample_centers(out['input_resized']).squeeze()
        out.update({'phosphene_brightness': simulator.get_state()['brightness'].squeeze(),
                    'target_brightness': cfg['target_brightness_scale']*target_pixels})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_brightness', 'target_brightness'),
                          weight=cfg['regularization_weight'])

    loss_func = CompoundLoss([recon_loss, regul_loss])

    return forward, loss_func

def get_pipeline_coactivation_loss(cfg):  
    def forward(batch, models, cfg, to_cpu=False):
        """Forward pass of the model."""

        # unpack
        encoder = models['encoder']
        decoder = models['decoder']
        simulator = models['simulator']

        # Data manipulation
        image, _ = batch

        # Forward pass
        simulator.reset()
        stimulation = encoder(image)
        phosphenes = simulator(stimulation).unsqueeze(1)
        reconstruction = decoder(phosphenes)
        
        coactivation = models['interaction'](stimulation) # current leaking to neighbouring electrodes

        # Output dictionary
        out = {'input':  image * cfg['circular_mask'],
               'stimulation': stimulation,
               'phosphenes': phosphenes,
               'reconstruction': reconstruction * cfg['circular_mask'],
               'input_resized': resize(image * cfg['circular_mask'], cfg['SPVsize'])}
        
        # Target phosphene brightness is sampled pixels at centers of the phosphenes
        target_pixels = simulator.sample_centers(out['input_resized']).squeeze()
        out.update({'phosphene_brightness': simulator.get_state()['brightness'].squeeze(),
                    'target_brightness': cfg['target_brightness_scale']*target_pixels,
                    'coactivation': coactivation})

        if to_cpu:
            # Return a cpu-copy of the model output
            out = {k: v.detach().cpu().clone() for k, v in out.items()}
        return out

    recon_loss = LossTerm(name='reconstruction_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('reconstruction', 'input'),
                          weight=1 - cfg['regularization_weight'])

    regul_loss = LossTerm(name='regularization_loss',
                          func=torch.nn.MSELoss(),
                          arg_names=('phosphene_brightness', 'target_brightness'),
                          weight=cfg['regularization_weight'])
    
    coact_loss = LossTerm(name='coactivation_loss',
                      func= lambda x1, x2: torch.mean(x1*x2), # mean of product
                      arg_names=('stimulation','coactivation'),
                      weight=cfg['coact_loss_scale'])


    loss_func = CompoundLoss([recon_loss, regul_loss, coact_loss])

    return forward, loss_func