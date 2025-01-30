"""
qinco.py

Contains the core QINCo class used at training time for multi-step residual quantization
with an (optionally) implicit codebook or sub-step logic.

Typically, you would import and instantiate this class inside your QINCoWrapper,
and use the QINCoInference classes in a separate file (qinco_inference.py) for
optimized inference.

Author: Your Name
"""

import torch
import torch.nn as nn
from einops import rearrange

# If you have distance utils, import them
# from mesh_autoencoder.utils.distances import pairwise_distances, compute_batch_distances
# or other relevant helper functions

class QINCo(nn.Module):
    """
    QINCo - A multi-step residual quantization class.

    This class handles training-time logic: 
    - forward(..., step="train") -> (codes, xhat, losses)
    - optionally forward(..., step="encode"/"decode") if you want to reuse the same class for inference.
      (But typically you'll have a separate qinco_inference.py for faster production inference.)
    """

    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary of hyperparameters needed for QINCo.
                Example keys:
                  - _D : dimension of input vectors
                  - _M : number of quantization steps (residual layers)
                  - codebook_size : how many codes per step
                  - beam_size : how many candidates in sub-step or beam search
                  - etc.

        For demonstration, we assume these config keys exist:
          config['_D']            = embedding dimension
          config['_M']            = number of residual steps (num_quantizers)
          config['codebook_size'] = size of each step's codebook
          config['A']             = sub-step candidates (or None if no sub-step)
          config['B']             = beam size
        """
        super().__init__()

        self.config = config
        self._D = config["_D"]
        self.M = config["_M"]
        self.codebook_size = config["codebook_size"]
        self.beam_size = config.get("B", 1)
        self.substep_size = config.get("A", 0)  # optional sub-step

        # Example: create M codebooks, each [codebook_size, D]
        # In a real design, you might add M submodules, each with an MLP or other logic
        self.steps = nn.ModuleList([])
        for i in range(self.M):
            step_module = QINCoStep(config, step_index=i)
            self.steps.append(step_module)

    def forward(self, x_in: torch.Tensor, step="train"):
        """
        The main forward method for training or optional encode/decode.

        If step=="train":
          Args:
            x_in: shape [B*N, D] 
              (you typically flatten [B, N, D] -> [B*N, D] before calling QINCo)

          Returns:
            codes: shape [M, B*N]
            xhat:  shape [B*N, D]
            losses: dict containing relevant losses (like MSE, sub-step losses, etc.)
        
        If step=="encode":
          Might return codes, xhat for inference-lke usage
        If step=="decode":
          x_in is codes -> return xhat
        """
        if step == "train":
            return self._train_forward(x_in)
        elif step == "encode":
            return self._encode(x_in)
        elif step == "decode":
            return self._decode(x_in)
        else:
            raise ValueError(f"Unknown step={step}. Must be one of ['train','encode','decode']")

    def _train_forward(self, x_in: torch.Tensor):
        """
        Training-time forward pass.
        Now includes commit loss in addition to MSE.
        """
        Bn, D = x_in.shape
        assert D == self._D
    
        codes_list = []
        residual = x_in.clone()
    
        commit_loss = 0.0  # Initialize commit loss
        for i, step_module in enumerate(self.steps):
            # Each step quantizes the residual
            codes_i, quant_i = step_module(residual)  # quant_i shape: [Bn, D], codes_i: [Bn]
            
            codes_list.append(codes_i)
    
            # Commit loss: encourage residuals to match quantized embeddings
            commit_loss_i = (residual.detach() - quant_i).pow(2).mean()  # Stop gradient on residual
            commit_loss += commit_loss_i
    
            # Update residual by subtracting the quantized vector
            residual = residual - quant_i
    
        # Final reconstruction
        xhat = x_in - residual  # Reconstruct x_in from residuals
    
        # Compute reconstruction loss (MSE)
        mse_loss = ((xhat - x_in) ** 2).mean()
    
        # Losses dictionary
        losses = {
            "mse_loss": mse_loss,
            "commit_loss": commit_loss,
        }
    
        # Stack codes: shape [M, Bn]
        codes = torch.stack(codes_list, dim=0)
    
        return codes, xhat, losses 

    def _encode(self, x_in: torch.Tensor):
        """
        Optional "encode" method.
        Typically might do the same as _train_forward except no backprop, 
        or might skip sub-step expansions (like beam search).
        """
        with torch.no_grad():
            codes, xhat, _ = self._train_forward(x_in)
        return codes, xhat

    def _decode(self, codes: torch.Tensor):
        """
        Optional "decode" method:
          codes: shape [M, B*N]
          Return xhat: shape [B*N, D]
        """
        M, BN = codes.shape
        # We'll reconstruct by summing code vectors from each step
        # Start from zero
        xhat = torch.zeros((BN, self._D), device=codes.device)
        for i, step_module in enumerate(self.steps):
            codes_i = codes[i]  # shape [BN]
            quant_i = step_module.get_code_vectors(codes_i)  # shape [BN, D]
            xhat = xhat + quant_i
        return xhat


class QINCoStep(nn.Module):
    """
    A single step of QINCo:
      - Has a codebook of shape [codebook_size, D].
      - At training time, picks the nearest code (or does sub-step logic).
      - At decode time, returns the code vectors from the discrete code indices.
    """

    def __init__(self, config, step_index):
        super().__init__()
        self.config = config
        self._D = config["_D"]
        self.codebook_size = config["codebook_size"]
        self.step_index = step_index

        # "Implicit" or "explicit" codebook. 
        # For demonstration, we store an explicit parameter of shape [codebook_size, D].
        self.codebook = nn.Parameter(
            torch.randn(self.codebook_size, self._D) * 0.01
        )

        # Optionally you can add small MLP or sub-step logic

    def forward(self, residual: torch.Tensor):
        """
        Single-step quantization at training time:
          1) For each sample in residual, find nearest code in self.codebook
          2) Return (codes_i, code_vectors)
        
        residual: shape [Bn, D]

        Returns:
          codes_i: shape [Bn]
          quant_i: shape [Bn, D]
        """
        Bn, D = residual.shape
        # naive nearest-neighbor approach
        # broadcast: [Bn, 1, D] - [1, codebook_size, D] => [Bn, codebook_size, D]
        diff = residual.unsqueeze(1) - self.codebook.unsqueeze(0)  
        dist = (diff ** 2).sum(dim=-1)  # [Bn, codebook_size]
        codes_i = dist.argmin(dim=-1)   # [Bn]
        # gather the codebook vectors
        quant_i = self.codebook[codes_i]  # shape [Bn, D]
        return codes_i, quant_i

    def get_code_vectors(self, codes_i: torch.Tensor):
        """
        Given discrete code indices, return the code vectors
        for reconstruction or inference.

        codes_i: shape [Bn]
        returns: shape [Bn, D]
        """
        return self.codebook[codes_i]


