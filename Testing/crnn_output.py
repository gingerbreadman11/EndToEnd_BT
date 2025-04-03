import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from local_datasets import MNIST_Sequence, get_mnist_dataset
import yaml
from utils import undo_standardize, resize

def visualize_sequence_and_phosphenes(sequence, encoder, simulator, interaction_model, save_path=None, batch_size=4):
    """Visualize input sequence and resulting phosphenes"""
    encoder.eval()
    
    # Create figure
    fig, axes = plt.subplots(2, sequence.shape[1], figsize=(15, 6))
    
    # Plot original sequence
    for i in range(sequence.shape[1]):
        axes[0, i].imshow(sequence[0, i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Input Frame {i+1}')
    
    with torch.no_grad():
        simulator.reset()
        
        # Forward pass - this is where the error is
        stimulation = encoder(sequence)
        print(f"Raw stimulation shape: {stimulation.shape}")
        
        # Handle batch size mismatch - replicate to match simulator's expected batch size
        if stimulation.shape[0] != batch_size:
            stimulation = stimulation.repeat(batch_size, 1)
            print(f"Adjusted stimulation shape: {stimulation.shape}")
        
        if interaction_model is not None:
            stimulation = interaction_model(stimulation).clip(min=0)
        
        # Now simulate 
        phosphenes = simulator(stimulation)
        print(f"Phosphenes shape: {phosphenes.shape}")
        
        # Take the first batch item for visualization
        phosphene_img = phosphenes[0].cpu().numpy()
        vmax = np.max(phosphene_img)
        vmin = np.min(phosphene_img)
    
    # Plot phosphenes
    for i in range(sequence.shape[1]):
        im = axes[1, i].imshow(phosphene_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Phosphene Output')
    
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return phosphenes

def main():
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load your configuration
    config_path = os.path.join(root_dir, '_config', 'alex_config', 'crnn_1.yaml')
    with open(config_path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    
    # Create a flattened config with all parameters
    cfg = {}
    # Add values from all sections
    sections = ['general', 'e2e_models', 'simulator', 'optimization', 'training_pipeline', 'dataset']
    for section in sections:
        if section in cfg_yaml:
            cfg.update(cfg_yaml[section])
    
    # Ensure device is set
    if 'device' not in cfg:
        cfg['device'] = 'cpu'
    
    # Set sequence length for MNIST
    cfg['sequence_length'] = 3
    
    # Get dataset
    trainset, valset = get_mnist_dataset(cfg)
    
    # Create models
    from init_training import get_models
    models = get_models(cfg)
    encoder = models['encoder']
    simulator = models['simulator']
    interaction_model = models.get('interaction', None)
    
    # Load trained model
    encoder_path = os.path.join(root_dir, 'Data_Storage', 'CRNN', 'checkpoints', 'best_encoder.pth')
    state_dict = torch.load(encoder_path)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    
    # Create output directory
    output_dir = os.path.join(root_dir, 'Testing', 'phosphene_outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the batch size from config
    batch_size = cfg.get('batch_size', 4)
    print(f"Using batch size: {batch_size}")
    
    # Process several test examples
    n_examples = 5
    for i in range(n_examples):
        sequence, label = valset[i]
        sequence = sequence.unsqueeze(0)  # Add batch dimension
        print(f"\nProcessing sequence {i+1} - Label: {label}")
        print(f"Input sequence shape: {sequence.shape}")
        
        # Visualize and save
        save_path = os.path.join(output_dir, f'phosphene_sequence_example_{i}.png')
        phosphenes = visualize_sequence_and_phosphenes(sequence, encoder, simulator, interaction_model, save_path, batch_size)

if __name__ == "__main__":
    main()