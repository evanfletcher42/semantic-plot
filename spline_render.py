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
    img_grad_mag = np.hypot(*np.gradient(img))

    # threshold gradients at gradient image mean: don't want high contrast objects getting all the lines
    img_grad_mag = np.minimum(img_grad_mag, np.mean(img_grad_mag) + np.std(img_grad_mag) * 0.5)

    # lines darken images; make them more likely to be in places where the image is dark
    img_blur = cv2.GaussianBlur(img, ksize=None, sigmaX=7)
    img_blur = img_blur / np.max(img_blur) * 0.5
    img_grad_mag *= (1.0 - img_blur)

    # Normalize pdf
    img_grad_mag /= np.sum(img_grad_mag)

    # plt.imshow(img_grad_mag)
    # plt.show()

    return img_grad_mag


def compute_pdf_semantic(imgi, loss_func):
    """
    Computes a map of spline-goes-here probabilities based on feature map gradients.

    This computes the *image* gradient comparing the current image against the target.
    Pixels with positive gradients are those where darkening the image (by adding splines) will improve the loss.

    We also require that these are co-located with actual edges in the image, and on sections that need ink.

    :param img: The current image, torch.Tensor, shape (h, w)
    :param loss_func: The semantic loss function instance to use here.
    :return: spline-goes-here probabilities, shape (h, w)
    """
    img = imgi.detach()

    with torch.enable_grad():
        img.requires_grad = True
        loss = loss_func(img)
        loss.backward()
        xg = img.grad.cpu().numpy().squeeze()

    # blur this a bit. This is going to have checkerboard-y artifacts that we don't want to draw.
    xg = cv2.GaussianBlur(xg, ksize=None, sigmaX=1.414)
    xg = np.clip(xg, a_min=0.0, a_max=None) # we care only about areas where the image wants to be darker

    # init lines where there are actual edges in the image
    img_grad_mag = np.hypot(*np.gradient(loss_func.target_img.cpu().numpy().squeeze()))
    img_grad_mag = np.minimum(img_grad_mag, np.mean(img_grad_mag) + np.std(img_grad_mag) * 0.5)

    # lines darken images; make them more likely to be in places where the image is dark
    target_np = loss_func.target_img.cpu().numpy().squeeze()
    img_blur = cv2.GaussianBlur(target_np, ksize=None, sigmaX=7)
    img_blur /= np.max(img_blur)

    xg = xg * img_grad_mag * (1.0 - img_blur)

    # Normalize pdf
    xg /= np.sum(xg)

    # plt.imshow(xg)
    # plt.show()
    # exit(0)

    return xg



class QuadraticSplineParams(nn.Module):
    """
    Contains and returns one set of parameters for a spline renderer.
    """
    def __init__(self, n_lines=64, img_shape=(512, 512), init_img=None):
        super().__init__()
        self.n_lines = n_lines
        self.a = nn.Parameter(torch.zeros(size=(1, n_lines, 2)), requires_grad=True)  # Quadratic spline start point
        self.b = nn.Parameter(torch.zeros(size=(1, n_lines, 2)), requires_grad=True)  # Quadratic spline control point
        self.c = nn.Parameter(torch.zeros(size=(1, n_lines, 2)), requires_grad=True)  # Quadratic spline end point
        self.lw = nn.Parameter(torch.ones(size=(1, n_lines, 1), requires_grad=True) * (0.7071 / img_shape[0]))  # Line weight
        self.lc = nn.Parameter(torch.ones(size=(1, n_lines, 1), requires_grad=True) * 0.5)  # Line color (intensity)

    def reinit_spline_split(self, replace_idx):
        """
        Re-initializes a spline by splitting the longest spline in half.

        :param replace_idx: Index of the spline to replace.
        """
        with torch.no_grad():
            # find longest, where length is very roughly approximated by distance from endpoints to control point.
            longest_idx = 0
            longest_len = 0.0
            for i in range(self.n_lines):
                # Candidate for splitting
                if i == replace_idx:
                    continue

                if self.lc[0, i, 0].item() < 0.025:
                    continue

                l = torch.linalg.norm(self.a[0, i, :] - self.b[0, i, :]) + torch.linalg.norm(self.b[0, i, :] - self.c[0, i, :]).item()
                if l > longest_len:
                    longest_idx = i
                    longest_len = l

        # split at midpoint
        t0 = 0.5

        # new control points
        b0 = (1 - t0) * self.a[0, longest_idx, :] + t0 * self.b[0, longest_idx, :]
        b1 = (1 - t0) * self.b[0, longest_idx, :] + t0 * self.c[0, longest_idx, :]

        # split point
        s0 = (1 - t0) * b0 + t0 * b1

        # modify splines
        self.a[0, replace_idx, :] = s0
        self.b[0, replace_idx, :] = b1
        self.c[0, replace_idx, :] = self.c[0, longest_idx, :]
        self.lw[0, replace_idx, :] = self.lw[0, longest_idx, :]
        self.lc[0, replace_idx, :] = self.lc[0, longest_idx, :]

        self.b[0, longest_idx, :] = b0
        self.c[0, longest_idx, :] = s0

        print(f"Split {longest_idx} (len {longest_len})")

    def init_lines(self, loss_func=None):
        """
        Initializes lines based on a semantic target, or randomly if there isn't one.
        :param loss_func: Cached loss function containing the semantic target.
        """
        with torch.no_grad():
            if loss_func is None:
                # random init
                for i in range(self.n_lines):
                    self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_random(self.a.device)
            else:
                ones = torch.ones(loss_func.target_img.shape, dtype=torch.float32, device=self.a.device)
                img_pdf = compute_pdf_semantic(ones, loss_func)
                img_cdf = np.cumsum(img_pdf)
                img_cdf = img_cdf / img_cdf[-1]
                for i in range(self.n_lines):
                    self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_pmap(self.a.device, img_cdf, img_pdf.shape)

    def reinit_invisible(self, min_intensity=0.025, curr_img=None, loss_func=None):
        with torch.no_grad():

            if curr_img is None:
                curr_img = torch.ones(loss_func.target_img.shape, dtype=torch.float32, device=self.a.device)

            img_pdf = compute_pdf_semantic(curr_img, loss_func)

            img_cdf = np.cumsum(img_pdf)
            img_cdf = img_cdf / img_cdf[-1]

            for i in range(self.n_lines):
                if self.lc[0, i, 0].item() < min_intensity:
                    print("reinit", i)

                    # Split long splines most of the time; reroll location sometimes.
                    if np.random.rand(1) < 0.80:
                        self.reinit_spline_split(i)
                    else:
                        # choose somewhere the probability map wants to
                        self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_pmap(self.a.device, img_cdf,
                                                                                                 img_pdf.shape)
                        self.lc[0, i, 0] = 0.5

    def save_svg(self, svg_path, img_sz=(512, 512)):
        with torch.no_grad():
            svg_data = f'<svg width="{img_sz[1]}" height="{img_sz[0]}" xmlns="http://www.w3.org/2000/svg">\n'

            for i in range(self.n_lines):
                a = self.a[0, i, :].cpu().numpy() * img_sz[0]
                b = self.b[0, i, :].cpu().numpy() * img_sz[0]
                c = self.c[0, i, :].cpu().numpy() * img_sz[0]
                gray = int(np.clip(255 * (1 - self.lc[0, i, 0].item()) + 0.5, 0, 255))
                lw = img_sz[0] * self.lw[0, i, 0].item()

                svg_data += f'  <path d="M{a[1]},{a[0]} Q{b[1]},{b[0]} {c[1]},{c[0]}" stroke="rgb({gray}, {gray}, {gray})" stroke-width="{lw}" fill="transparent"/>\n'

            svg_data += '</svg>'

            with open(svg_path, 'w') as svg_file:
                svg_file.write(svg_data)

    def forward(self):
        # Return parameters directly
        return self.a, self.b, self.c, self.lw, self.lc


class QuadraticSplineRenderer(nn.Module):
    """
    A self-contained stroke renderer for quadratic splines.
    """

    # TODO: Efficiency improvements.
    #  - This uses a clean, but still fairly complex, analytical solution for quadratic spline distance.
    #    It may be more efficient to render a spline as a set of line segments.
    #  - This will result in a dense gradient computation for every line and pixel in the image.  Many pixels will have
    #    zero gradient wrt. parameters.  If there were a way to take advantage of this sparsity, that would be _amazing_

    def __init__(self, img_shape=(256, 256)):
        super().__init__()
        self.img_shape = img_shape

        # Texture coordinates
        # Register these as a buffer so .to(device) works
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, self.img_shape[1]),
                                torch.linspace(0.0, 1.0, self.img_shape[0]),
                                indexing='ij')
        self.register_buffer("p", torch.stack([yy, xx], dim=-1), persistent=False)

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

        # This is a simple guard against point b being at (0, 0), which would make this blow up.
        b = b + 1e-5

        # reshapes for broadcasting

        p_orig_shape = p.shape
        p = p.reshape((1, 1, -1, 2))  # To shape: (..., 2)

        a = a[:, :, None, :]
        b = b[:, :, None, :]
        c = c[:, :, None, :]

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

        d1 = (D + B * T[:, :, :, 0, None]) * T[:, :, :, 0, None] - C
        d2 = (D + B * T[:, :, :, 1, None]) * T[:, :, :, 1, None] - C

        dd1 = torch.sum(d1 * d1, dim=-1)
        dd2 = torch.sum(d2 * d2, dim=-1)

        d = safe_sqrt(torch.minimum(dd1, dd2))
        d = d.reshape((d.shape[0], d.shape[1], 1, *p_orig_shape[:-1]))

        return d

    def forward(self, a, b, c, lw, lc):

        # dist shape: (n, 1, rows, cols)
        dist = self.quadratic_bezier_distance(p=self.p, a=a, b=b, c=c)

        # note: these are white (1.0) lines on a dark (0.0) background
        intensity = torch.sigmoid((lw[..., None, None] - dist) / lw[..., None, None] * 6.0) * lc[..., None, None]

        # Additive blending + clamp over all lines --> shape (rows, cols, 1)
        # TODO: Actual blend should be: blended = intensity + (1 - intensity) * prev_img
        #       That can be done in a loop.  I probably need to be reminded why doing it this way is a bad idea.
        intensity = torch.sum(intensity, dim=1)

        # Invert (we want dark lines on white background)
        intensity = 1 - intensity
        return intensity


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchinfo
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    line_params = QuadraticSplineParams(n_lines=768, img_shape=(256, 256)).to(device)
    line_params.init_lines()

    lines = QuadraticSplineRenderer(img_shape=(256, 256)).to(device)

    # with torch.no_grad():
    for i in range(1):
        t1 = time.perf_counter_ns()
        img = lines(*line_params())
        t2 = time.perf_counter_ns()
        img_mean = torch.mean(img)
        img_mean.backward()
        t3 = time.perf_counter_ns()
        print("Render time:", (t2 - t1) * 1e-6, "ms")
        print("Backprop time:", (t3 - t2) * 1e-6, "ms")
        print("max mem:", torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        print("img shape:", img.shape)

    img_n = img.detach().cpu().numpy()[0, 0, ...]
    plt.imshow(img_n, cmap='gray')
    plt.show()
