import random
import numpy as np
import traceback
from tqdm.auto import tqdm
import wandb
import torch
import math
torch.pi = math.pi
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.utils import make_grid, save_image
from torchvision import transforms

# import matplotlib.pyplot as plt
from pathlib import Path


def fix_experiment_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

fix_experiment_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
print ("Device: ", device)

# Helper Functions
def show_image(image, nrow=8):
  # Input: image
  # Displays the image using matplotlib
  #grid_img = make_grid(image.detach().cpu(), nrow=nrow, padding=0)
  #plt.imshow(grid_img.permute(1, 2, 0))
  #plt.axis('off')
  pass


def get_dataloaders(data_root, batch_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize))
    
    train = datasets.SVHN(data_root, split='train', download=True, transform=transform)
    test  = datasets.SVHN(data_root, split='test', download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader

 
# Visualize the Dataset
def visualize():
  train_dataloader, _ = get_dataloaders(data_root=data_root, batch_size=train_batch_size)
  imgs, labels = next(iter(train_dataloader))

  save_image((imgs + 1.) * 0.5, './results/orig.png')
  show_image((imgs + 1.) * 0.5)

# if __name__ == '__main__':
#   visualize()

 
# ## Define the Model Architectures
class Encoder(nn.Module):
  def __init__(self, nc, nef, nz, isize, device):
    super(Encoder, self).__init__()

    # Device
    self.device = device

    # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
    self.encoder = nn.Sequential(
      nn.Conv2d(nc, nef, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef),

      nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 2),

      nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 4),

      nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 8),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    hidden = self.encoder(inputs)
    hidden = hidden.view(batch_size, -1)
    return hidden

class Decoder(nn.Module):
  def __init__(self, nc, ndf, nz, isize):
    super(Decoder, self).__init__()

    # Map the latent vector to the feature map space
    self.ndf = ndf
    self.out_size = isize // 16
    self.decoder_dense = nn.Sequential(
      nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
      nn.ReLU(True)
    ).to(device)

    self.decoder_conv = nn.Sequential(
      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf, nc, 3, 1, padding=1)
    ).to(device)

  def forward(self, input):
    batch_size = input.size(0)
    hidden = self.decoder_dense(input).view(
      batch_size, self.ndf * 8, self.out_size, self.out_size).to(device)
    output = self.decoder_conv(hidden).to(device)
    return output

 
class DiagonalGaussianDistribution(object):
  # Gaussian Distribution with diagonal covariance matrix
  def __init__(self, mean, logvar=None):
    super(DiagonalGaussianDistribution, self).__init__()
    # Parameters:
    # mean: A tensor representing the mean of the distribution
    # logvar: Optional tensor representing the log of the standard variance
    #         for each of the dimensions of the distribution 

    self.mean = mean.to(device)
    if logvar is None:
        logvar = torch.zeros_like(self.mean).to(device)
    self.logvar = torch.clamp(logvar, -30., 20.)
    self.batch_size = self.mean.size(0)
    self.latent_size = self.mean.view(self.batch_size,-1).shape[1]
    self.std = torch.exp(0.5 * self.logvar).to(device)
    self.var = torch.exp(self.logvar).to(device)

  def sample(self):
    # Provide a reparameterized sample from the distribution
    # Return: Tensor of the same size as the mean
    sample = self.mean.to(device) + self.std.to(device) * torch.normal(mean=torch.zeros_like(self.std), std=torch.ones_like(self.std)).to(device)
    return sample

  def kl(self):
    # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
    # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch

    # kl_loss = nn.KLDivLoss(reduction="batchmean")
    # kl_div = kl_loss(self.sample(), torch.normal(mean=torch.zeros_like(self.std), std=torch.ones_like(self.std)))
    try:
        kl_div = 0.5 * (-torch.sum(torch.log(self.var), dim=-1) - self.latent_size + torch.sum(self.var, dim=-1) + torch.sum((self.mean**2), dim=-1))
    except:
        traceback.print_exc()
    return kl_div

  def nll(self, sample, dims=[1, 2, 3]):
    # Computes the negative log likelihood of the sample under the given distribution
    # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
    # negative_ll = None    # WRITE CODE HERE
    negative_ll =  0.5 * (self.latent_size * torch.log(torch.tensor(torch.pi * 2))
                            + torch.sum(self.logvar, dim=dims)
                            + torch.sum((sample-self.mean)**2 / self.var, dim=dims)
                            ).to(device)
    return negative_ll

  def mode(self):
    # Returns the mode of the distribution
    mode = self.mean     # WRITE CODE HERE
    return mode

 
class VAE(nn.Module):
  def __init__(self, in_channels=3, decoder_features=32, encoder_features=32, z_dim=100, input_size=32, device=torch.device("cuda:0")):
    super(VAE, self).__init__()

    self.z_dim = z_dim
    self.in_channels = in_channels
    self.device = device

    # Encode the Input
    self.encoder = Encoder(nc=in_channels, 
                            nef=encoder_features, 
                            nz=z_dim, 
                            isize=input_size, 
                            device=device
                            )

    # Map the encoded feature map to the latent vector of mean, (log)variance
    out_size = input_size // 16
    self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim).to(device)
    self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim).to(device)

    # Decode the Latent Representation
    self.decoder = Decoder(nc=in_channels, 
                           ndf=decoder_features, 
                           nz=z_dim, 
                           isize=input_size
                           )

  def encode(self, x):
    # Input:
    #   x: Tensor of shape (batch_size, 3, 32, 32)
    # Returns:
    #   posterior: The posterior distribution q_\phi(z | x)

    # WRITE CODE HERE
    approx_posterior = self.encoder(x)
    return DiagonalGaussianDistribution(mean=self.mean(approx_posterior), logvar=self.logvar(approx_posterior))

  def decode(self, z):
    # Input:
    #   z: Tensor of shape (batch_size, z_dim)
    # Returns
    #   conditional distribution: The likelihood distribution p_\theta(x | z)
    
    # WRITE CODE HERE
    approx_posterior_x = self.decoder(z)
    return DiagonalGaussianDistribution(mean=approx_posterior_x)

  def sample(self, batch_size):
    # Input:
    #   batch_size: The number of samples to generate
    # Returns:
    #   samples: Generated samples using the decoder
    #            Size: (batch_size, 3, 32, 32)
    try:
        _z_shape = (batch_size, self.decoder.ndf)
        z = torch.normal(mean=torch.zeros(_z_shape), std=torch.ones(_z_shape)).to(self.device)
        posterior_x = self.decode(z)
    except:
        traceback.print_exc()
    return posterior_x.mode()
    # WRITE CODE HERE

    pass

  def log_likelihood(self, x, K=100):
    # Approximate the log-likelihood of the data using Importance Sampling
    # Inputs:
    #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
    #   K: Number of samples to use to approximate p_\theta(x)
    # Returns:
    #   ll: Log likelihood of the sample x in the VAE model using K samples
    #       Size: (batch_size,)
    posterior = self.encode(x)
    prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

    log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
    for i in range(K):
      z = posterior.sample()                        # WRITE CODE HERE (sample from q_phi)
      recon = self.decode(z)                    # WRITE CODE HERE (decode to conditional distribution)
      log_likelihood[:, i] = -(recon.nll(x, dims=[1, 2, 3]) + prior.nll(z, dims=[1]) - posterior.nll(z, dims=[1]))    # WRITE CODE HERE (log of the summation terms in approximate log-likelihood, that is, log p_\theta(x, z_i) - log q_\phi(z_i | x))
      del z, recon
    
    ll = torch.logsumexp(log_likelihood, dim=-1) - torch.log(torch.tensor(K))     # WRITE CODE HERE (compute the final log-likelihood using the log-sum-exp trick)
    return ll

  def forward(self, x):
    # Input:
    #   x: Tensor of shape (batch_size, 3, 32, 32)
    # Returns:
    #   reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
    #                   Size: (batch_size, 3, 32, 32)
    #   Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
    #                                         Size: (batch_size,)
    #   KL: The KL Divergence between the variational approximate posterior with N(0, I)
    #       Size: (batch_size,)
    posterior = self.encode(x)    # WRITE CODE HERE
    latent_z = posterior.sample()     # WRITE CODE HERE (sample a z)
    recon = self.decode(latent_z)        # WRITE CODE HERE (decode)

    return recon.mode(), recon.nll(x), posterior.kl()

 
def interpolate(model, z_1, z_2, n_samples):
  # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
  # Inputs:
  #   z_1: The first point in the latent space
  #   z_2: The second point in the latent space
  #   n_samples: Number of points interpolated
  # Returns:
  #   sample: The mode of the distribution obtained by decoding each point in the latent space
  #           Should be of size (n_samples, 3, 32, 32)
  lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(device)
  z = z_2 + lengths * (z_1 - z_2)    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
  return model.decode(z).mode()


if __name__ == '__main__':
  results_folder = Path("./results")
  results_folder.mkdir(exist_ok = True)

  # Training Hyperparameters
  train_batch_size = 64   # Batch Size
  z_dim = 32        # Latent Dimensionality
  lr = 1e-4         # Learning Rate

  # Define Dataset Statistics
  image_size = 32
  input_channels = 3
  data_root = '../data'
  epochs = 30
  exp_args = {"train_batch_size":train_batch_size,"z_dim":z_dim,"lr":lr,"image_size":image_size,"input_channels":input_channels,"data_root": data_root, "epochs": epochs, "device": str(device), "model": "VAE"}
  wandb.init(project="RepLearning - A3",config=exp_args)
  model = VAE(in_channels=input_channels, 
            input_size=image_size, 
            z_dim=z_dim, 
            decoder_features=32, 
            encoder_features=32, 
            device=device
            )
  model.to(device)
  optimizer = Adam(model.parameters(), lr=lr)
  train_dataloader, _ = get_dataloaders(data_root, batch_size=train_batch_size)
  for epoch in range(epochs):
    with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
      model.train()
      print("Epoch: ", epoch)
      for batch in tepoch:
        tepoch.set_description(f"Epoch: {epoch}")

        optimizer.zero_grad()

        imgs, _ = batch
        batch_size = imgs.shape[0]
        x = imgs.to(device)

        recon, nll, kl = model(x)
        loss = (nll + kl).mean()

        loss.backward()
        optimizer.step()
        wandb.log({"loss":loss.item(), "nll":nll.mean().item(), "kl":kl.mean().item()})
        tepoch.set_postfix(loss=loss.item())

    samples = model.sample(batch_size=64)
    save_image((x + 1.) * 0.5, './results/orig.png')
    save_image((recon + 1.) * 0.5, './results/recon.png')
    save_image((samples + 1.) * 0.5, f'./results/samples_{epoch}.png')

  show_image(((samples + 1.) * 0.5).clamp(0., 1.))
  
  # Once the training of the model is done, we can use the model to approximate the log-likelihood of the test data using the function that we defined above.
  _, test_dataloader = get_dataloaders(data_root, batch_size=train_batch_size)
  with torch.no_grad():
    with tqdm(test_dataloader, unit="batch", leave=True) as tepoch:
      model.eval()
      log_likelihood = 0.
      num_samples = 0.
      for batch in tepoch:
        tepoch.set_description(f"Epoch: {epoch}")
        imgs,_ = batch
        batch_size = imgs.shape[0]
        x = imgs.to(device)

        log_likelihood += model.log_likelihood(x).sum()
        num_samples += batch_size
        tepoch.set_postfix(log_likelihood=log_likelihood / num_samples)

  z_1 = torch.randn(1, z_dim).to(device)
  z_2 = torch.randn(1, z_dim).to(device)

  interp = interpolate(model, z_1, z_2, 10)
  show_image((interp + 1.) * 0.5, nrow=10)
  wandb.finish()