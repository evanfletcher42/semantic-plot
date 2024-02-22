import torch
import torch.nn as nn
import numpy as np
from math_helpers import cubrt, safe_acos, safe_sqrt


class QuadraticSplineRenderer(nn.Module):
    """
    A self-contained stroke renderer for quadratic splines.
    Contains parameters for a configurable number of lines, handles blending them into a monochrome image.
    """

    # TODO: Efficiency improvements.
    #  - This uses a clean, but still fairly complex, analytical solution for quadratic spline distance.
    #    It may be more efficient to render a spline as a set of line segments.
    #  - This will result in a dense gradient computation for every line and pixel in the image.  Many pixels will have
    #    zero gradient wrt. parameters.  If there were a way to take advantage of this sparsity, that would be _amazing_

    def __init__(self, n_lines=64, img_shape=(512, 512)):
        super().__init__()
        self.img_shape = img_shape

        self.a = nn.Parameter(torch.rand(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline start point
        self.b = nn.Parameter(torch.rand(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline control point
        self.c = nn.Parameter(torch.rand(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline end point

        self.lw = nn.Parameter(torch.ones(size=(n_lines, 1, 1, 1), requires_grad=True) * (0.7071 / img_shape[0]))  # Line weight
        self.lc = nn.Parameter(torch.ones(size=(n_lines, 1, 1, 1), requires_grad=True) * 0.5)  # Line color (intensity)

    def solve_cubic(self, ax, ay, az):
        p = ay - ax * ax / 3.0
        p3 = p * p * p
        q = ax * (2 * ax * ax - 9 * ay) / 27.0 + az
        d = q * q + 4.0 * p3 / 27.0

        # case d > 0:
        x0 = cubrt((1.0 * safe_sqrt(d) - q) * 0.5)
        x1 = cubrt((-1.0 * safe_sqrt(d) - q) * 0.5)

        root0 = x0 + x1 - ax / 3.0

        # case d <= 0:
        p3_sdiv = torch.where(abs(p3) > 1e-9, p3, torch.ones_like(p3)*1e-9)

        v = safe_acos(-safe_sqrt(-27.0/p3_sdiv)*q*0.5)/3.0
        m = torch.cos(v)
        n = torch.sin(v) * np.sqrt(3)

        root1 = (m + m) * safe_sqrt(-p/3.) - ax/3.
        root2 = (-n-m) * safe_sqrt(-p/3.) - ax/3.

        roots_single = torch.concat([root0, root0], dim=-1)
        roots_double = torch.concat([root1, root2], dim=-1)

        roots = torch.where(d > 0, roots_single, roots_double)

        return roots

    def quadratic_bezier_distance(self, p, a, b, c):
        """
        Determine the distance in two dimensions between points in p and the quadratic Bezier curve defined by a, b, c.

        The quadratic Bezier curve here is of the form:
        p0 * (1-t)^2 + p1 * 2t(1-t) + p2 * t^2
        where 0 <= t <= 1.

        See: https://www.shadertoy.com/view/lsdBDS

        :param p: 2D coordinates of test points.  Shape: (..., 2)
        :param a: 2D coordinates of the start point, at t=0.  Shape: (n, 2)
        :param b: 2D coordinates of the control point.  Shape: (n, 2)
        :param c: 2D coordinates of the end point, at t=1.  Shape: (n, 2)

        :return: Linear distance between p and the specified curve, shape (n, ..., 1)
        """

        # TODO: It looks like the original author(s) may have considerably simplified this.  Try that sometime.

        # TODO: This is a simple guard against point b being at (0, 0), which would make this blow up.
        b = b + 1e-5

        # reshapes for broadcasting

        p_orig_shape = p.shape
        p = p.reshape((1, -1, 2))  # To shape: (..., 2)

        a = a[:, None, :]
        b = b[:, None, :]
        c = c[:, None, :]

        A = b - a
        B = c - b - A
        C = p - a
        D = A * 2.0

        B_dot_B = torch.sum(B * B, dim=-1, keepdim=True)

        cubic_a = (-3.0 * torch.sum(A * B, dim=-1, keepdim=True)) / (-B_dot_B)
        cubic_b = (torch.sum(C * B, dim=-1, keepdim=True) - 2.0 * torch.sum(A * A, dim=-1, keepdim=True)) / (-B_dot_B)
        cubic_c = (torch.sum(C * A, dim=-1, keepdim=True)) / (-B_dot_B)

        roots = self.solve_cubic(cubic_a, cubic_b, cubic_c)

        # T = roots
        T = torch.clip(roots, 0.0, 1.0)

        d1 = (D + B * T[:, :, 0, None]) * T[:, :, 0, None] - C
        d2 = (D + B * T[:, :, 1, None]) * T[:, :, 1, None] - C

        dd1 = torch.sum(d1 * d1, dim=-1)
        dd2 = torch.sum(d2 * d2, dim=-1)

        d = safe_sqrt(torch.minimum(dd1, dd2))
        d = d.reshape((d.shape[0], *p_orig_shape[:-1], 1))

        return d

    def forward(self):  # Look ma, no 'x'
        # Texture coordinates (TBD: requires_grad ?)
        d = self.a.device

        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, self.img_shape[1], device=d),
                                torch.linspace(0.0, 1.0, self.img_shape[0], device=d),
                                indexing='ij')
        p = torch.stack([yy, xx], dim=-1)

        # dist shape: (n, rows, cols, 1)
        dist = self.quadratic_bezier_distance(p=p, a=self.a, b=self.b, c=self.c)

        # note: these are white (1.0) lines on a dark (0.0) background
        intensity = torch.sigmoid((self.lw - dist) / self.lw * 6.0) * self.lc

        # Additive blending + clamp over all lines --> shape (rows, cols, 1)
        # TODO: Actual blend should be: blended = intensity + (1 - intensity) * prev_img
        #       That can be done in a loop.  I probably need to be reminded why doing it this way is a bad idea.
        intensity = torch.clamp(torch.sum(intensity, dim=0), 0.0, 1.0)

        # Invert (we want dark lines on white background)
        intensity = 1 - intensity
        return intensity


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchinfo
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines = QuadraticSplineRenderer(n_lines=256, img_shape=(512, 512)).to(device)

    # with torch.no_grad():
    for i in range(1):
        t1 = time.perf_counter_ns()
        img = lines()
        t2 = time.perf_counter_ns()
        print("Render time:", (t2 - t1) * 1e-6, "ms")
        print("max mem:", torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    img_n = img.detach().cpu().numpy()
    plt.imshow(img_n, cmap='gray')
    plt.show()
