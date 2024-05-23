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


# assume (0,0) is the center
def get_ball(radius: float, n_interior_points: int, n_boundary_points: int, dim=2):
    def get_boundary_points(radius: float, n_points: int) -> torch.Tensor:
        from torch import pi, cos, sin
        points = torch.zeros(size=(n_points, dim))
        angles = pi * torch.rand(n_points)
        points[:, 0] = radius * cos(angles)
        points[:, 1] = radius * sin(angles)
        return points

    def get_interior_points(xmin: float, xmax: float, radius: float, n_points: int) -> torch.Tensor:
        inside_points = 0
        points = torch.zeros(size=(n_points, dim))
        while inside_points < n_points:
            candidate = ((xmax - xmin) * torch.rand(1, 2) +
                         xmin).reshape(-1, dim)
            if torch.linalg.norm(candidate) < radius:
                points[inside_points, :] = candidate
                inside_points += 1
        return points
    return get_interior_points(-radius, radius, radius, n_interior_points), \
        get_boundary_points(radius, n_boundary_points)
