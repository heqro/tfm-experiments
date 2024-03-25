import torch


def d_meshgrid(*args):
    # Create grid coordinates along each dimension
    grid = torch.meshgrid(*args)
    # Convert the grid coordinates to tensors
    grid_tensors = [coord.flatten() for coord in grid]
    # Stack the grid tensors along a new dimension to create the d-dimensional meshgrid
    meshgrid = torch.stack(grid_tensors, dim=-1)
    return meshgrid


def cheb_points(num: int):
    naturals_list = torch.arange(start=0, end=num, step=1)
    arg_cos = torch.pi * (2 * naturals_list + 1)/(2 * num)
    # take opposite to return sorted list
    return -torch.cos(arg_cos)
