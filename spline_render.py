import torch
import torch.nn as nn
import numpy as np
from math_helpers import cubrt, safe_acos, safe_sqrt
# import matplotlib.pyplot as plt
import cv2

def init_spline_random(device):
    """
    Initializes a single spline, as a small-ish, random stroke, with limited curvature, in a sensible location.
    :param device: Device for the output tensor
    :return: Tuple (start_pt, ctrl_pt, end_pt), each tensors on the specified device with shape (2).
    """

    min_line_len = 0.01
    max_line_len = 0.05

    line_len = np.random.uniform(min_line_len, max_line_len)
    line_angle = np.random.uniform(0.0, 2*np.pi)
    line_vec = np.array([np.cos(line_angle), np.sin(line_angle)]) * line_len

    start_pt = np.random.uniform(0.0 + line_len, 1.0 - line_len, size=2)
    end_pt = start_pt + line_vec
    ctrl_pt = (start_pt + end_pt) / 2 + np.random.normal(size=2) * line_len / 2

    return torch.from_numpy(start_pt).type(torch.float32).to(device), \
           torch.from_numpy(ctrl_pt).type(torch.float32).to(device), \
           torch.from_numpy(end_pt).type(torch.float32).to(device)

def init_spline_pmap(device, img_cdf, img_shape):
    """
    Initializes a single spline, contingent on the provided image-space cdf

    :param device: Device for the output tensor
    :param img_cdf: 2d cumulative probability distribution
    :return: Tuple (start_pt, ctrl_pt, end_pt), each tensors on the specified device with shape (2).
    """
    x = np.random.rand()
    idx = np.searchsorted(img_cdf, x)
    pt = np.unravel_index(idx, img_shape)
    ctr_pt = np.array([pt[0]/img_shape[0], pt[1]/img_shape[1]])

    min_line_len = 0.01
    max_line_len = 0.05

    line_len = np.random.uniform(min_line_len, max_line_len)
    line_angle = np.random.uniform(0.0, 2 * np.pi)
    line_vec = np.array([np.cos(line_angle), np.sin(line_angle)]) * line_len

    start_pt = ctr_pt - line_vec / 2
    end_pt = ctr_pt + line_vec / 2
    ctrl_pt = (start_pt + end_pt) / 2 + np.random.normal(size=2) * line_len / 2

    return torch.from_numpy(start_pt).type(torch.float32).to(device), \
           torch.from_numpy(ctrl_pt).type(torch.float32).to(device), \
           torch.from_numpy(end_pt).type(torch.float32).to(device)

def compute_pdf_grads(img):
    """
    Converts an image into a map of spline-goes-here probabilities based on gradient magnitude.
    :param img: Image to compute a map from, shape (h, w)
    :return: spline-goes-here probabilities, shape (h, w)
    """
    img_blur = cv2.GaussianBlur(img, ksize=None, sigmaX=7)
    img_blur = img_blur / np.max(img_blur)

    img_grad_mag = np.hypot(*np.gradient(img)) * (1.0 - img_blur)
    img_grad_mag /= np.sum(img_grad_mag)

    # plt.imshow(img_grad_mag)
    # plt.show()

    return img_grad_mag


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

    def __init__(self, n_lines=64, img_shape=(512, 512), init_img=None):
        super().__init__()
        self.img_shape = img_shape

        self.a = nn.Parameter(torch.zeros(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline start point
        self.b = nn.Parameter(torch.zeros(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline control point
        self.c = nn.Parameter(torch.zeros(size=(n_lines, 2)), requires_grad=True)  # Quadratic spline end point

        with torch.no_grad():
            if init_img is None:
                # random init
                for i in range(n_lines):
                    self.a[i, :], self.b[i, :], self.c[i, :] = init_spline_random(self.a.device)
            else:
                # image-contingent init
                img_pdf = compute_pdf_grads(init_img)
                img_cdf = np.cumsum(img_pdf)
                img_cdf = img_cdf / img_cdf[-1]
                for i in range(n_lines):
                    self.a[i, :], self.b[i, :], self.c[i, :] = init_spline_pmap(self.a.device, img_cdf, img_pdf.shape)

        self.lw = nn.Parameter(torch.ones(size=(n_lines, 1, 1, 1), requires_grad=True) * (0.7071 / img_shape[0]))  # Line weight
        self.lc = nn.Parameter(torch.ones(size=(n_lines, 1, 1, 1), requires_grad=True) * 0.5)  # Line color (intensity)

    def solve_cubic(self, ax, ay, az):
        p = ay - ax * ax / 3.0
        p3 = p * p * p
        q = ax * (2 * ax * ax - 9 * ay) / 27.0 + az
        d = q * q + 4.0 * p3 / 27.0

        # case d > 0: one root
        x0 = cubrt((1.0 * safe_sqrt(d) - q) * 0.5)
        x1 = cubrt((-1.0 * safe_sqrt(d) - q) * 0.5)

        root0 = x0 + x1 - ax / 3.0

        # case d <= 0: technically 3 roots, but the center one can't be the closest so we'll ignore it.
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
        T = torch.clamp(roots, 0.0, 1.0)

        d1 = (D + B * T[:, :, 0, None]) * T[:, :, 0, None] - C
        d2 = (D + B * T[:, :, 1, None]) * T[:, :, 1, None] - C

        dd1 = torch.sum(d1 * d1, dim=-1)
        dd2 = torch.sum(d2 * d2, dim=-1)

        d = safe_sqrt(torch.minimum(dd1, dd2))
        d = d.reshape((d.shape[0], 1, *p_orig_shape[:-1]))

        return d

    def forward(self):  # Look ma, no 'x'
        # Texture coordinates (TBD: requires_grad ?)
        d = self.a.device

        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, self.img_shape[1], device=d),
                                torch.linspace(0.0, 1.0, self.img_shape[0], device=d),
                                indexing='ij')
        p = torch.stack([yy, xx], dim=-1)

        # dist shape: (n, 1, rows, cols)
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

    lines = QuadraticSplineRenderer(n_lines=768, img_shape=(256, 256)).to(device)

    # with torch.no_grad():
    for i in range(1):
        t1 = time.perf_counter_ns()
        img = lines()
        t2 = time.perf_counter_ns()
        print("Render time:", (t2 - t1) * 1e-6, "ms")
        print("max mem:", torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    img_n = img.detach().cpu().numpy()[0, ...]
    plt.imshow(img_n, cmap='gray')
    plt.show()
