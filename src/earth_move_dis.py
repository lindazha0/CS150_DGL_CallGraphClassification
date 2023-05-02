# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
from scipy.stats import wasserstein_distance
import torch


def sliced_wasserstein_distance(x, y, num_projections=1000):
    """
    Compute the Sliced Wasserstein Distance between two tensors, or SWD for short.
    args:
        x, y: tensors
        num_projections: number of random projections to use
    """
    # Check if tensors have the same length
    assert x.shape[0] == y.shape[0], "Input tensors must have the same length"
    
    # Generate random projections
    projections = torch.randn(num_projections, x.shape[0])
    
    # Project the input tensors onto the random projections
    x_proj = torch.matmul(projections, x)
    y_proj = torch.matmul(projections, y)
    
    # Calculate the Wasserstein distance for each projection and average the results
    swd = torch.mean(torch.abs(x_proj - y_proj))
    
    return swd


def emd_distance(V1, V2):
    """
    Compute the Earth Mover's Distance between two vectors,
    which are the embeddings of two graphs
    However, not apply to Tensors since they call numpy functions inside
    """
    print(f"input types: {type(V1)}, {type(V2)}")
    print(f"input types: {V1}, {V2}")
    return wasserstein_distance(V1, V2)


def main():
    V1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    V2 = torch.tensor([2.0, 6.0, 3.0], requires_grad=True)

    print(f"input types: {type(V1)}, {type(V2)}")
    distance = sliced_wasserstein_distance(V1, V2)
    print(f"Sliced Wasserstein Distance: {distance}")
    return

if __name__ == "__main__":
    main()