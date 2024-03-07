import torch
from grid_utils import d_meshgrid


def plot_figure(centers, fn, nn, dim: int, path: str = 'output'):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with torch.no_grad():
        x = d_meshgrid(*[torch.linspace(-2, 2, 300)
                         for _ in range(dim)]).reshape(-1, dim).to('cuda')
        fig = plt.figure()
        target_points = fn(x)
        approx_points = nn(x)
        center_points = fn(centers)
        plt.plot(x.cpu(), target_points.cpu(), label='Target')
        plt.plot(x.cpu(), approx_points.cpu(), label='Approx')
        plt.scatter(centers.cpu(), center_points.cpu())
        plt.legend()
        pp = PdfPages(f'{path}.pdf')
        pp.savefig(fig, bbox_inches='tight')
        pp.close()
        plt.close()


def plot_figure_3d(centers, fn, nn, dim: int, res=60, path: str = 'output'):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with torch.no_grad():
        min_coord, max_coord = -0.2, 1.2
        x = d_meshgrid(*[torch.linspace(0, 1, res)
                       for _ in range(dim)]).reshape(-1, dim).to('cuda')
        approx_points = nn(x).cpu()
        target_points = fn(x).reshape_as(approx_points).cpu()
        fig = plt.figure()
        plt.imshow((target_points - approx_points).abs().reshape((res, res)), extent=(
            min_coord, max_coord, min_coord, max_coord), cmap="hot", norm="log")
        plt.gca().add_patch(Rectangle((0, 0), 1, 1, edgecolor='green', facecolor="none"))
        plt.colorbar()
        plt.scatter(centers[:, 0].cpu(), centers[:, 1].cpu())
        pp = PdfPages(f'{path}.pdf')
        pp.savefig(fig, bbox_inches='tight')
        pp.close()
        plt.close()
