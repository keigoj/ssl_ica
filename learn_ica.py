from transformers import Wav2Vec2Model, Wav2Vec2Processor
from dataset_from_manifest import DatasetFromManifest
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from logging import FileHandler
from dataclasses import dataclass
<<<<<<< HEAD
from typing import Union
=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
import torch.distributed as dist
import torch
import numpy as np
import logging
import random
import os
import argparse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%X]",
    handlers=[FileHandler(filename="log.txt", mode="w")],
)
random.seed(2024)
torch.random.manual_seed(2024)
np.random.seed(2024)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


@dataclass
class ClusterEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    
    @property
    def is_master(self):
        return self.rank == 0
    
    @property
    def is_multigpu(self):
        return self.world_size > 1
    
    def __del__(self):
        destroy_cluster()
    
    def all_reduce_sum(self, data):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        
    def all_reduce_sum_scalar(self, data, dtype):
        data = torch.tensor(data, dtype=dtype, device=self.device)
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        return data.item()
    
    def broadcast(self, data: torch.Tensor, src=0):
        dist.barrier()
        data = data.contiguous()
        dist.broadcast(data, src=src)    
    
    def all_gather_objects(self, data):
        outputs = [None for _ in range(self.world_size)]
        dist.all_gather_object(outputs, data)
        return outputs

    def gather_objects(self, data):
        outputs = [None for _ in range(self.world_size)]
        dist.gather_object(data, outputs if self.is_master else None, dst=0)
        return outputs
    
    def barrier(self):
        dist.barrier()

def destroy_cluster():
    if "OMPI_COMM_WORLD_SIZE" not in os.environ:
        return

def batch_idx_fn(batch):
    waves, texts = list(zip(*batch))
    waves = [np.array(wave) for wave in waves]
    inputs = processor(waves, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)

    return inputs, texts

def dump_ssl_feature(dataset_path, save_dir, percent=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DatasetFromManifest(dataset_path)
    n_samples = len(train_dataset)
    indices = list(range(n_samples))
    random_indices = random.sample(indices, int(n_samples*percent))
    sampler = SubsetRandomSampler(random_indices)
    dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=16, num_workers=4, collate_fn=batch_idx_fn)

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    model.to(device)

    all_hidden_states = []
    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            input_value = batch[0].to(device)
            outputs = model(input_value.input_values, attention_mask=input_value.attention_mask, output_hidden_states=True, return_dict=True)

            hidden_states = torch.stack(outputs.hidden_states[1:]) # n_layers x n_batch x n_frames x n_features
            l, b, t, d = hidden_states.shape
            hidden_states = hidden_states.view(l, b*t, d).cpu() # n_layers x (n_batch x n_frames) x n_features
            all_hidden_states.append(hidden_states)

            del input_value, hidden_states
            torch.cuda.empty_cache()
    
    all_hidden_states = torch.cat(all_hidden_states, dim=1) # n_layers x (n_sample x n_frames) x n_features

    for i, states in enumerate(tqdm(all_hidden_states)):
        file_path = f"{save_dir}/layer_{i+1}.pt"
        torch.save(states, file_path)


### PCA
@torch.no_grad()
def torch_pca(X: torch.Tensor, n_components: int, batch_size: int = 10000, segment_device: str = "cuda"):
    """PCA for PyTorch tensor.
    
    Args:
        X: torch.Tensor, shape (n_samples, n_features)
            Input data.
        n_components: int
            Number of components to keep.
        batch_size: int, default=10000
            Batch size for computing the data covariance matrix.

    Returns:
        trans_mat_from_right: torch.Tensor, shape (n_features, n_components)
            The transformation matrix.
        X_mean: torch.Tensor, shape (1, n_features)
            The mean of the input data.
    """
    n_samples, n_features = X.shape
    if batch_size > 0:
        X_mean = torch.zeros(1, n_features, device=segment_device)
        for i in range(0, n_samples, batch_size):
            X_mean += X[i:i+batch_size,:].to(segment_device).sum(dim=0, keepdim=True)
        X_mean /= n_samples
        cov = torch.zeros(n_features, n_features, device=segment_device)
        for i in range(0, n_samples, batch_size):
            target_X = X[i:i+batch_size,:].to(segment_device) - X_mean
            cov += target_X.T @ target_X
        cov = cov / n_samples
        del X
        torch.cuda.synchronize()
    else:
        X_mean = X.mean(dim=0, keepdim=True)
        X -= X_mean # n_samples x n_features
        cov = (X.T @ X)/n_samples # n_features x n_features   
        del X
    U, S, _ = torch.svd(cov) # U: n_features x n_features, S: n_features
    W = (U @ torch.diag(1/S.sqrt())).T # n_features x n_features
    trans_mat_from_right = W[:n_components,:].T # n_features x n_components
    
    return trans_mat_from_right, X_mean

# ICA
@torch.no_grad()
<<<<<<< HEAD
def torch_fdica(X: torch.Tensor, n_iterations: int, initial_W: torch.Tensor = None, cluster_env: ClusterEnv=None, eps: float = 1e-3): # eps 1e-10
=======
def torch_fdica(X: torch.Tensor, n_iterations: int, initial_W: torch.Tensor = None, eps: float = 1e-3):
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    """FastICA for PyTorch tensor.
    
    Args:
        X: torch.Tensor, shape (n_frequencies, n_samples, n_components)
            Input data.
        n_iterations: int
            Number of iterations

    Returns:
        W: torch.Tensor, shape (n_frequencies, n_components x n_components)
            Demixing matrices
        Y: torch.Tensor, shape (n_frequencies, n_samples x n_components)
            Demixed features
    """
<<<<<<< HEAD
    is_multigpu = (cluster_env is not None) and cluster_env.is_multigpu

    n_frequencies, n_samples, n_components = X.shape[0], X.shape[1], X.shape[2]

    if is_multigpu:
        n_total_samples = cluster_env.all_reduce_sum_scalar(n_samples, dtype=torch.int64)
        n_samples = n_total_samples

=======

    n_frequencies, n_samples, n_components = X.shape[0], X.shape[1], X.shape[2]
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    logging.info(f"n_frequencies: {n_frequencies}, n_samples: {n_samples}, n_components: {n_components}")
    
    def get_loss(W: torch.Tensor, X: torch.Tensor):
        '''Compute loss
        
        Args:
            W (torch.Tensor): n_frequencies x n_components x n_components
            X (torch.Tensor): n_frequencies x n_samples x n_components
            
        Returns:
            loss (torch.Tensor): scalar
        '''
        Y = X @ W.transpose(1,2) # (n_frequencies x n_samples x n_components) x (n_frequencies x n_components x n_components) -> n_frequencies x n_samples x n_components
        Y_abs_sum = Y.abs().sum() # 1
        
        return Y_abs_sum - n_samples*torch.slogdet(W)[1].sum()
    
    if initial_W is not None:
        W = initial_W.clone()
<<<<<<< HEAD
        if is_multigpu:
            cluster_env.broadcast(W)
=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    else:
        W = torch.eye(n_components, dtype=X.dtype, device=X.device)[None,:,:].repeat(n_frequencies, 1, 1) # n_frequencies x n_components x n_components
    I = torch.eye(n_components, dtype=X.dtype, device=X.device)[None,:,:].repeat(n_frequencies, 1, 1) # n_frequencies x n_components x n_components
    loss_list = [get_loss(W, X).item()]

    for iteration in tqdm(range(n_iterations)):
        logging.info(f"Iteration {iteration + 1}/{n_iterations}, Loss: {loss_list[-1]}")
        for k in range(n_components):
            reciprocal_R_k = torch.einsum('ftm,fm->ft', X, W[:, k, :]) # n_frequencies x n_samples
            reciprocal_R_k.abs_() # n_frequencies x n_samples
            reciprocal_R_k.clamp_min_(eps) # n_frequencies x n_samples
            reciprocal_R_k.reciprocal_() # n_frequencies x n_samples
            V_k = torch.einsum('ftm,ft,ftn->fmn', X, reciprocal_R_k, X)/n_samples # n_frequencies x n_components x n_components
<<<<<<< HEAD
            if is_multigpu:
                cluster_env.all_reduce_sum(V_k)
=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
            W[:, k, :] = torch.linalg.solve(W @ V_k, I[:, :, k]) # n_frequencies x n_components x 1
            W[:, k, :] *= (torch.einsum('fm,fmn,fn->f', W[:, k, :], V_k, W[:, k, :]) + eps).rsqrt()[:,None]
        loss_list.append(get_loss(W, X).item())
    Y = X @ W.transpose(1,2)
    
    return W, Y, {"loss_list": np.array(loss_list)}

<<<<<<< HEAD
# IVA
@torch.no_grad()
def torch_iva(X: torch.Tensor, n_iterations: int=100, initial_W: torch.Tensor=None, W_update_algo: str="IP", cluster_env: ClusterEnv=None, eps: float=1e-8) -> Union[torch.Tensor, torch.Tensor, dict]:
    '''Perform independent vector analysis (IVA) with iterative projection
    
    Args:
        X (torch.Tensor): Features (n_frequencies x n_samples x n_features)
        n_iterations (int): Number of iterations

    Returns:
        W (torch.Tensor): Demixing matrices (n_frequencies x n_features x n_features)
        Y (torch.Tensor): Demixed features (n_frequencies x n_samples x n_features)
    '''
    is_multigpu = (cluster_env is not None) and cluster_env.is_multigpu
    
    n_frequencies, n_samples, n_features = X.shape[0], X.shape[1], X.shape[2]

    if is_multigpu:
        n_total_samples = cluster_env.all_reduce_sum_scalar(n_samples, dtype=torch.int64)
        n_samples = n_total_samples

    def get_loss(W: torch.Tensor, Y: torch.Tensor):
        '''Compute loss
        
        Args:
            W (torch.Tensor): n_frequencies x n_features x n_features
            X (torch.Tensor): n_frequencies x n_samples x n_features
            
        Returns:
            loss (torch.Tensor): scalar
        '''
        Y_norm_sum = torch.linalg.norm(Y, dim=0).sum()
        if is_multigpu:
            cluster_env.all_reduce_sum(Y_norm_sum)
        return Y_norm_sum - n_samples*torch.slogdet(W)[1].sum()

    # Set initial demixing matrices
    if initial_W is not None:
        W = initial_W.clone()
        if is_multigpu:
            cluster_env.broadcast(W)
    else:
        W = torch.eye(n_features, dtype=X.dtype, device=X.device)[None,:,:].repeat(n_frequencies, 1, 1) # n_frequencies x n_features x n_features
    
    I = torch.eye(n_features, dtype=X.dtype, device=X.device)[None,:,:].repeat(n_frequencies, 1, 1) # n_frequencies x n_features x n_features

    # Initialize Y
    Y = X @ W.transpose(1,2) # n_frequencies x n_samples x n_features
    # Initialize loss list
    loss_list = [get_loss(W, Y).item()]
    # Main loop
    for iteration in range(n_iterations):
        if (not is_multigpu) or (is_multigpu and cluster_env.is_master):
            logging.info(f"Iteration {iteration}/{n_iterations}, Loss: {loss_list[-1]}")
        # Compute the inverse of the variance
        reciprocal_R = torch.linalg.norm(Y, dim=0) # n_samples x n_features
        reciprocal_R.clamp_min_(eps)
        reciprocal_R.reciprocal_()
        
        # Update the demixing matrix
        if W_update_algo == "IP":
            for k in range(n_features):
                V_k = torch.einsum('ftm,t,ftn->fmn', X, reciprocal_R[:,k], X)/n_samples # n_frequencies x n_features x n_features
                if is_multigpu:
                    cluster_env.all_reduce_sum(V_k)
                W[:, k, :] = torch.linalg.solve(W @ V_k, I[:, :, k]) # n_frequencies x n_features x 1
                W[:, k, :] *= (torch.einsum('fm,fmn,fn->f', W[:, k, :], V_k, W[:, k, :]) + eps).rsqrt()[:,None]
        else:
            raise NotImplementedError(f"Update algorithm {W_update_algo} is not implemented.")
        # Update Y
        Y = X @ W.transpose(1,2)
        # Compute loss
        loss_list.append(get_loss(W,Y).item())
    return W, Y, {"loss_list": np.array(loss_list)}

=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
def learn_icamagaxis_by_torch(
    ssl_feat_dir,
    ica_path,
    n_clusters,
    max_iter,
    batch_size,
    n_layers=12,
    device="cpu",
):
    if not torch.cuda.is_available() and device == "cuda":
        logging.warning("CUDA is not available. Using CPU instead.")
        device = "cpu"
    device = torch.device(device)
    
    pca_trans_list = []
    mean_list = []
    pca_results = []
    
    for i in tqdm(range(0, n_layers)):
        feat = torch.load(f"{ssl_feat_dir}/layer_{i+1}.pt") # n_sample x n_features

        if feat.shape[1] < n_clusters:
            raise ValueError(f"n_clusters must be less than the number of features: {feat.shape[1]}")

        feat = feat.to(device)
        trans_mat_from_right, mean = torch_pca(feat, n_clusters, batch_size=batch_size, segment_device="cuda" if torch.cuda.is_available() else "cpu")
        X = feat.to(device)
        X = (X - mean) @ trans_mat_from_right
        if n_clusters is not None:
            if X.shape[1] != n_clusters:
                raise ValueError(f"Number of components in PCA result and n_components are different: {X.shape[1]} != {n_clusters}")
        pca_trans_list.append(trans_mat_from_right)
        mean_list.append(mean)
        pca_results.append(X)
        del feat, trans_mat_from_right, mean, X
        torch.cuda.empty_cache()

    X = torch.stack(pca_results) # n_frequencies x n_samples x n_components
    W, Y, ica_history = torch_fdica(X, n_iterations=max_iter)
    del X, Y, ica_history
    torch.cuda.empty_cache()

    trans_mat_from_right = torch.stack(pca_trans_list).cpu() # n_frequencies x n_features x n_components
    mean = torch.stack(mean_list).cpu() # n_frequencies x 1 x n_features
    W = W.cpu()

    torch.save({"trans_mat_from_right": trans_mat_from_right, "mean": mean, "W": W}, ica_path)
    logging.info("finished successfully")

<<<<<<< HEAD
def learn_iva_by_torch(
    ssl_feat_dir,
    ica_path,
    iva_path,
    n_clusters,
    max_iter,
    batch_size,
    n_layers=12,
    device="cpu",
):
    if not torch.cuda.is_available() and device == "cuda":
        logging.warning("CUDA is not available. Using CPU instead.")
        device = "cpu"
    device = torch.device(device)

    pca_trans_list = []
    mean_list = []
    pca_results = []

    data = torch.load(ica_path)
    pca_w = data["trans_mat_from_right"]
    mean = data["mean"]
    ica_w = data["W"]

    for i in tqdm(range(0, n_layers)):
        feat = torch.load(f"{ssl_feat_dir}/layer_{i+1}.pt") # n_sample x n_features

        if feat.shape[1] < n_clusters:
            raise ValueError(f"n_clusters must be less than the number of features: {feat.shape[1]}")

        feat = feat.to(device)
        # trans_mat_from_right, mean = torch_pca(feat, n_clusters, batch_size=batch_size, segment_device="cuda" if torch.cuda.is_available() else "cpu")
        mean = mean[i].to(device)
        trans_mat_from_right = pca_w[i].to(device)

        X = feat.to(device)
        X = (X - mean) @ trans_mat_from_right
        if n_clusters is not None:
            if X.shape[1] != n_clusters:
                raise ValueError(f"Number of components in PCA result and n_components are different: {X.shape[1]} != {n_clusters}")
        pca_trans_list.append(trans_mat_from_right)
        mean_list.append(mean)
        pca_results.append(X)
        del feat, trans_mat_from_right, mean, X
        torch.cuda.empty_cache()

    X = torch.stack(pca_results) # n_frequencies x n_samples x n_components
    W, Y, ica_history = torch_iva(X, initial_W=ica_w ,n_iterations=max_iter)
    del X, Y, ica_history
    torch.cuda.empty_cache()

    trans_mat_from_right = torch.stack(pca_trans_list).cpu() # n_frequencies x n_features x n_components
    mean = torch.stack(mean_list).cpu() # n_frequencies x 1 x n_features
    W = W.cpu()

    torch.save({"trans_mat_from_right": trans_mat_from_right, "mean": mean, "W": W}, iva_path)
    logging.info("finished successfully")

=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--percent", type=float)
    parser.add_argument("--nclusters", type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    train_path = "data/train-clean-100-manifest.json"
<<<<<<< HEAD
    percent = 0.05
    ssl_feat_dir = f"data/ssl_features/subset{percent}"
    n_clusters = 100
    method = "iva"
=======
    percent = 0.01
    ssl_feat_dir = f"data/ssl_features/subset{percent}"
    n_clusters = 300
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2

    if not os.path.exists(f"{ssl_feat_dir}/layer_1.pt"):
        dump_ssl_feature(train_path, ssl_feat_dir, percent)
    
<<<<<<< HEAD
    if method == "ica":
        learn_icamagaxis_by_torch(
            ssl_feat_dir=ssl_feat_dir,
            ica_path=f"model/iva/iva_{n_clusters}.pt", 
            n_clusters=n_clusters, 
            max_iter=100, 
            batch_size=10000, 
            device="cuda"
        )
    elif method == "iva":
        learn_iva_by_torch(
            ssl_feat_dir=ssl_feat_dir,
            ica_path=f"model/ica/ica_{n_clusters}.pt",
            iva_path=f"model/iva/iva_{n_clusters}.pt", 
            n_clusters=n_clusters, 
            max_iter=100, 
            batch_size=10000, 
            device="cuda"
        )
    else:
        raise ValueError(f"Not implimentation: {method}")
=======
    learn_icamagaxis_by_torch(
        ssl_feat_dir=ssl_feat_dir,
        ica_path=f"model/ica/ica_{n_clusters}.pt", 
        n_clusters=n_clusters, 
        max_iter=100, 
        batch_size=10000, 
        device="cuda"
    )
    
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2

if __name__ == "__main__":
    main()
