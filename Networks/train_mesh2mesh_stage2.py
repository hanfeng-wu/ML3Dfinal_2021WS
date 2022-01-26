from base64 import decode
from pathlib import Path
from Networks.mesh2mesh import *
import torch
from Networks.pointclouddataset import *
from chamferdist import ChamferDistance





def train(encoder_pretrained ,encoder, trainloader, valloader, device, config):

    # TODO Declare loss and move to specified device
    loss_criterion = torch.nn.MSELoss()

    # TODO Declare optimizer
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}],lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    encoder.train()


    # TODO Implement the training loop. It looks very much the same as in the previous exercise part, except that you are now using points instead of voxel grids

    # keep track of best validation accuracy achieved so that we can save the weights
    best_val = 1000000000.

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):
        for i, batch in enumerate(trainloader):
            # move batch to device

            partial = batch['partial'].to(device)
            full = batch['full'].to(device)
            partial = partial.permute(0,2,1)
            full = full.permute(0,2,1)

            # TODO: zero out previously accumulated gradients
            optimizer.zero_grad()

            #print(batch['points'])

            # TODO: forward pass
            #print(batch)
            latent_vector = encoder(partial)
            latent_vector_gt = encoder_pretrained(full)

            

            # TODO: compute total loss = sum of loss for whole prediction + losses for partial predictions
            #loss_total = torch.zeros([1], dtype=batch['points'].dtype, requires_grad=True).to(device)
            # TODO: Loss due to prediction[:, output_idx, :] (output_idx=0 for global prediction, 1-8 local)
                
                
                
            loss_total = loss_criterion(latent_vector, latent_vector_gt).mean()

            # TODO: compute gradients on loss_total (backward pass)
            loss_total.mean().backward()

            # TODO: update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.mean().item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.10f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                encoder.eval()

                loss_total_val = 0
                # forward pass and evaluation for entire validation set
                for batch in valloader:
                    partial = batch['partial'].to(device)
                    full = batch['full'].to(device)
                    partial = partial.permute(0,2,1)
                    full = full.permute(0,2,1)

                    with torch.no_grad():
                        # TODO: Get prediction scores
                        latent_vector = encoder(partial)
                        latent_vector_gt = encoder_pretrained(full)
                        loss_total_val += loss_criterion(latent_vector, latent_vector_gt).mean()
                    

                    # TODO: keep track of total / correct / loss_total_val

                

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(valloader):.10f}')

                if loss_total_val < best_val:
                    torch.save(encoder.state_dict(), f'./models/runs/{config["experiment_name"]}/encoder_stage2_best.pth')
                    best_val = loss_total_val

                # set model back to train
                encoder.train()


def main(config):
    """
    Function for training PointNet on ShapeNet
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # Create Dataloaders
    train_dataset = pcd_stage2(split='train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )
    val_dataset = pcd_stage2(split='val' if not config['is_overfit'] else 'overfit')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )    

    # Instantiate model
    encoder_pretrained = PointNetEncoder(numpoints=4000)
    assert config['encoder_pretrained'] is not None
    encoder_pretrained.load_state_dict(torch.load(config['encoder_pretrained'], map_location='cpu'))
    encoder_pretrained.eval()
    encoder = PointNetEncoder(numpoints=1000)

    # Load model if resuming from checkpoint
    if config['resume_ckpt_en'] is not None:
        encoder.load_state_dict(torch.load(config['resume_ckpt_en'], map_location='cpu'))

    # Move model to specified device
    encoder.to(device)
    encoder_pretrained.to(device)

    #freeze the encoder pretrained parameters
    for param in encoder_pretrained.parameters():
        param.requires_grad = False

    # Create folder for saving checkpoints
    Path(f'models/runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)

    # Start training
    train(encoder_pretrained ,encoder, train_dataloader, val_dataloader, device, config)