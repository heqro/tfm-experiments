from typing import Literal
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
def get_ball(radius: float, n_interior_points: int, n_boundary_points: int, dim=2, requires_grad=False):
    def get_boundary_points(radius: float, n_points: int) -> torch.Tensor:
        from torch import pi, cos, sin
        points = torch.zeros(size=(n_points, dim))
        angles = 2 * pi * torch.rand(n_points)
        points[:, 0] = radius * cos(angles)
        points[:, 1] = radius * sin(angles)
        points.requires_grad = requires_grad
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
        points.requires_grad = requires_grad
        return points
    return get_interior_points(-radius, radius, radius, n_interior_points), \
        get_boundary_points(radius, n_boundary_points)


def get_rectangle(bottom_left: tuple[float, float], width: float, height: float,
                  n_interior_points: int, n_points_per_boundary: int, requires_grad=False, separate_into_sets=False) \
        -> dict[Literal['boundary', 'interior', 'corners'], torch.Tensor | dict[Literal['left', 'right', 'top', 'bottom'], torch.Tensor] |
                dict[Literal['top_left', 'bottom_left', 'top_right', 'bottom_right'], torch.Tensor]] | torch.Tensor:
    if width <= 0 or height <= 0 or n_interior_points <= 0 or n_points_per_boundary <= 0:
        raise Exception('Parameters should be positive')
    x = bottom_left[0] * torch.ones(n_points_per_boundary).reshape(-1, 1)
    y = height * \
        torch.rand(n_points_per_boundary).reshape(-1, 1) + bottom_left[1]
    left_bd = torch.cat((x, y), dim=1)
    left_bd.requires_grad = requires_grad

    x = (bottom_left[0] + width) * \
        torch.ones(n_points_per_boundary).reshape(-1, 1)
    y = height * \
        torch.rand(n_points_per_boundary).reshape(-1, 1) + bottom_left[1]
    right_bd = torch.cat((x, y), dim=1)
    right_bd.requires_grad = requires_grad

    x = width * torch.rand(n_points_per_boundary).reshape(-1,
                                                          1) + bottom_left[0]
    y = (bottom_left[1] + height) * \
        torch.ones(n_points_per_boundary).reshape(-1, 1)
    top_bd = torch.cat((x, y), dim=1)
    top_bd.requires_grad = requires_grad

    x = width * torch.rand(n_points_per_boundary).reshape(-1,
                                                          1) + bottom_left[0]
    y = bottom_left[1] * \
        torch.ones(n_points_per_boundary).reshape(-1, 1)
    bottom_bd = torch.cat((x, y), dim=1)
    bottom_bd.requires_grad = requires_grad

    top_left_corner = torch.tensor(
        [bottom_left[0], bottom_left[1]+height], requires_grad=requires_grad).reshape(-1, 2)
    top_right_corner = torch.tensor(
        [bottom_left[0]+width, bottom_left[1]+height], requires_grad=requires_grad).reshape(-1, 2)
    bottom_left_corner = torch.tensor(
        [bottom_left[0], bottom_left[1]], requires_grad=requires_grad).reshape(-1, 2)
    bottom_right_corner = torch.tensor(
        [bottom_left[0]+width, bottom_left[1]], requires_grad=requires_grad).reshape(-1, 2)

    # interior = torch.rand((n_interior_points, 2), requires_grad=requires_grad)
    interior_x = width * \
        torch.rand(n_interior_points,
                   requires_grad=requires_grad).reshape(-1, 1) + bottom_left[0]
    interior_y = height * \
        torch.rand(n_interior_points,
                   requires_grad=requires_grad).reshape(-1, 1) + bottom_left[1]
    interior = torch.cat((interior_x, interior_y), dim=1)
    if separate_into_sets:
        return {'boundary': {
            'left': left_bd,
            'right': right_bd,
            'top': top_bd,
            'bottom': bottom_bd
        },
            'interior': interior,
            'corners': {
            'top_left': top_left_corner,
            'top_right': top_right_corner,
            'bottom_left': bottom_left_corner,
            'bottom_right': bottom_right_corner
        }}
    else:
        return torch.cat((interior, top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, left_bd, right_bd, top_bd, bottom_bd), dim=0)
