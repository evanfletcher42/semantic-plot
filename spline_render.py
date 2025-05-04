import torch
import torch.nn as nn
import numpy as np
from math_helpers import cubrt, safe_acos, safe_sqrt
# import matplotlib.pyplot as plt
import cv2
import tqdm
import xml.etree.ElementTree as ET


def init_spline_random(device, img_shape):
    """
    Initializes a single spline, as a small-ish, random stroke, with limited curvature, in a sensible location.
    :param device: Device for the output tensor
    :param img_shape: Shape of target image (for aspect ratio)
    :return: Tuple (start_pt, ctrl_pt, end_pt), each tensors on the specified device with shape (2).
    """

    min_line_len = 0.01
    max_line_len = 0.05

    lim_y = img_shape[0] / min(*img_shape)
    lim_x = img_shape[1] / min(*img_shape)
    lims = np.array([lim_y, lim_x])

    line_len = np.random.uniform(min_line_len, max_line_len)
    line_angle = np.random.uniform(0.0, 2*np.pi)
    line_vec = np.array([np.cos(line_angle), np.sin(line_angle)]) * line_len

    start_pt = np.random.uniform(0.0 + line_len, lims - line_len, size=2)
    end_pt = start_pt + line_vec
    ctrl_pt = (start_pt + end_pt) / 2 + np.random.normal(size=2) * line_len / 2

    return torch.from_numpy(start_pt).type(torch.float32).to(device), \
           torch.from_numpy(ctrl_pt).type(torch.float32).to(device), \
           torch.from_numpy(end_pt).type(torch.float32).to(device)


def img_eigenvectors(img, ctr):
    row, col = ctr
    radius = 4
    r_start, r_end = max(row - radius, 0), min(row + radius + 1, img.shape[0]-1)
    c_start, c_end = max(col - radius, 0), min(col + radius + 1, img.shape[1]-1)
    roi = img[r_start:r_end, c_start:c_end]

    rows, cols = np.arange(r_start, r_end) - row, np.arange(c_start, c_end) - col
    dr, dc = np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=0)
    mr = np.sum(roi * dr) / np.sum(roi)
    mc = np.sum(roi * dc) / np.sum(roi)
    Srr = np.sum(roi * dr ** 2)
    Src = np.sum(roi * dr * dc)
    Scc = np.sum(roi * dc ** 2)
    cov = np.array([[Srr, Src], [Src, Scc]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    mean = np.array([mr, mc]) + ctr
    principal = eigvecs[:, np.argmax(eigvals)]
    secondary = eigvecs[:, np.argmin(eigvals)]

    # plt.imshow(roi, extent=[c_start - col, c_end - col, r_end - row, r_start - row])
    # plt.scatter(mc, mr, marker='+', color='red')
    # plt.arrow(mc, mr, principal[1], principal[0])
    # plt.show()

    return mean, principal, secondary


def init_spline_pmap(device, img_pdf, img_cdf, img_shape):
    """
    Initializes a single spline, contingent on the provided image-space cdf

    :param device: Device for the output tensor
    :param img_pdf: 2d probability distribution
    :param img_cdf: 2d cumulative probability distribution
    :param img_shape: Shape of target
    :return: Tuple (start_pt, ctrl_pt, end_pt), each tensors on the specified device with shape (2).
    """
    # Roll against the image pdf
    x = np.random.rand()
    idx = np.searchsorted(img_cdf, x)
    pt = np.unravel_index(idx, img_shape)

    pscale = 1.0 / min(*img_shape)

    ctr_pt = np.array([pt[0] * pscale, pt[1] * pscale])

    # Greedily choose the biggest point
    # pt = np.unravel_index(np.argmax(img_pdf), img_shape)
    # ctr_pt = np.array([pt[0]/img_shape[0], pt[1]/img_shape[1]])
    # print(ctr_pt)

    # Crop out a ROI near this point and see how anisotropic it is.
    mean, principal, secondary = img_eigenvectors(img_pdf, pt)

    # Use these to draw a spline through the found point.
    # line_len = 0.030
    line_len = 0.015
    line_vec = principal * line_len

    start_pt = ctr_pt - line_vec / 2
    end_pt = ctr_pt + line_vec / 2
    ctrl_pt = np.array([mean[0] * pscale, mean[1] * pscale])

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
    img_np = img.cpu().numpy().squeeze()

    with torch.enable_grad():
        img.requires_grad = True
        loss = loss_func(img)
        loss.backward()
        xg = img.grad.cpu().numpy().squeeze()

    # blur this a bit. This is going to have checkerboard-y artifacts that we don't want to draw.
    xg = cv2.GaussianBlur(xg, ksize=None, sigmaX=2.82)
    xg = np.clip(xg, a_min=0.0, a_max=None) # we care only about areas where the image wants to be darker

    # init lines where there are actual edges in the image
    img_grad_mag = np.hypot(*np.gradient(loss_func.target_img.cpu().numpy().squeeze()))
    img_grad_mag = np.minimum(img_grad_mag, np.mean(img_grad_mag) + np.std(img_grad_mag) * 0.5)

    # lines darken images; make them more likely to be in places where the image is dark
    target_np = loss_func.target_img.cpu().numpy().squeeze()
    img_blur = cv2.GaussianBlur(target_np, ksize=None, sigmaX=7)
    img_blur /= np.max(img_blur)

    # multiply by image: there's not much point drawing more lines where there are already lines, no matter what grads say
    xg = xg * img_grad_mag * (1.0 - img_blur) * img_np
    xg = np.clip(xg, a_min=0.0, a_max=None)

    # Normalize pdf
    xg /= np.sum(xg)

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
        self.lw = nn.Parameter(torch.ones(size=(1, n_lines, 1), requires_grad=True) * (1.0 / min(*img_shape)))  # Line weight
        self.lc = nn.Parameter(torch.ones(size=(1, n_lines, 1), requires_grad=True) * 0.5)  # Line color (intensity)
        self.img_shape = img_shape
        self.min_intensity = 0.005

    def reinit_spline_split(self, replace_idx):
        """
        Re-initializes a spline by splitting the longest or most bent spline.

        :param replace_idx: Index of the spline to replace.
        """
        with torch.no_grad():
            # Extract spline parameters
            a = self.a[0]       # Shape: (n_lines, 2)
            b = self.b[0]       # Shape: (n_lines, 2)
            c = self.c[0]       # Shape: (n_lines, 2)
            lc = self.lc[0, :, 0]  # Shape: (n_lines,)

            # Mask off the spline we are replacing, and any others that are probably about to be replaced
            indices = torch.arange(self.n_lines, device=lc.device)
            mask = (indices != replace_idx) & (lc >= self.min_intensity)

            if not mask.any():
                print("No suitable spline to split.")
                return

            ab = a - b
            bc = b - c
            lengths = torch.norm(ab, dim=1) + torch.norm(bc, dim=1)  # Shape: (n_lines,)

            # compute "foldedness" (ratio of length vs distance between endpoints)
            # Note the minus-1. Straight lines have a folded-ness of 0. Anything more bent is a bigger number.
            ac = a - c
            ac_norm = torch.norm(ac, dim=1) + 1e-5
            foldedness = (lengths / ac_norm) - 1.0  # Shape: (n_lines,)

            badness = lengths * foldedness

            badness_masked = badness.clone()
            badness_masked[~mask] = -float('inf')

            max_badness, max_idx = torch.max(badness_masked, dim=0)

            if max_badness == -float('inf'):
                print("No suitable spline to split.")
                return

            # Split the selected spline at t0 = 0.5 (midpoint)
            t0 = 0.5
            a_max = a[max_idx]
            b_max = b[max_idx]
            c_max = c[max_idx]

            # Compute new control points
            b0 = (1 - t0) * a_max + t0 * b_max
            b1 = (1 - t0) * b_max + t0 * c_max
            s0 = (1 - t0) * b0 + t0 * b1

            # Update the replace_idx spline with the new split spline
            self.a[0, replace_idx, :] = s0
            self.b[0, replace_idx, :] = b1
            self.c[0, replace_idx, :] = c_max
            self.lw[0, replace_idx, :] = self.lw[0, max_idx, :]
            self.lc[0, replace_idx, :] = self.lc[0, max_idx, :]

            # Update the original max_idx spline to reflect the split
            self.b[0, max_idx, :] = b0
            self.c[0, max_idx, :] = s0

            # Print debug information
            print(f"Split spline {max_idx} (badness {max_badness.item():.4f}, "
                  f"length {lengths[max_idx].item():.4f}, fold {foldedness[max_idx].item():.4f})")

    def init_lines(self, loss_func=None, renderer=None):
        """
        Initializes lines based on a semantic target, or randomly if there isn't one.
        :param loss_func: Cached loss function containing the semantic target.
        """
        with torch.no_grad():
            if loss_func is None:
                # random init
                for i in range(self.n_lines):
                    self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_random(self.a.device, self.img_shape)
            else:
                # lines with zero intensity do not affect the image
                self.lc.fill_(0)

                # iteratively draw on the image and add lines as gradients demand
                eval_n = self.n_lines // 30
                for i in tqdm.tqdm(range(self.n_lines), desc="Iterative Init Lines"):

                    # periodically reevaluate the pdf
                    if i % eval_n == 0:
                        drawn = renderer(*self.forward())
                        img_pdf = compute_pdf_semantic(drawn, loss_func)
                        img_cdf = np.cumsum(img_pdf)
                        img_cdf = img_cdf / img_cdf[-1]

                    self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_pmap(self.a.device, img_pdf, img_cdf, img_pdf.shape)

                    # init relatively light
                    self.lc[0, i, 0] = 0.10

    def reinit_invisible(self, curr_img=None, loss_func=None):
        with torch.no_grad():
            ab = self.a[0] - self.b[0]
            bc = self.b[0] - self.c[0]
            lengths = (torch.norm(ab, dim=1) + torch.norm(bc, dim=1)).cpu().numpy()  # Shape: (n_lines,)

            for i in range(self.n_lines):
                if self.lc[0, i, 0].item() < self.min_intensity or lengths[i] < 1.0 / max(*self.img_shape):

                    # Split long splines most of the time; reroll location sometimes.
                    if np.random.rand(1) < 0.80:
                        self.reinit_spline_split(i)
                    else:
                        # choose somewhere the probability map wants to
                        if curr_img is None:
                            curr_img = torch.ones(loss_func.target_img.shape, dtype=torch.float32, device=self.a.device)

                        img_pdf = compute_pdf_semantic(curr_img, loss_func)

                        img_cdf = np.cumsum(img_pdf)
                        img_cdf = img_cdf / img_cdf[-1]

                        self.a[0, i, :], self.b[0, i, :], self.c[0, i, :] = init_spline_pmap(self.a.device, img_pdf, img_cdf,
                                                                                                 img_pdf.shape)
                        self.lc[0, i, 0] = 0.10
                        print("reinit", i, "by replacement")

    def save_svg(self, svg_path, img_sz=(512, 512)):
        with torch.no_grad():
            svg_data = f'<svg width="{img_sz[1]}" height="{img_sz[0]}" xmlns="http://www.w3.org/2000/svg">\n'
            sc = min(*img_sz)
            an = self.a.cpu().numpy()
            bn = self.b.cpu().numpy()
            cn = self.c.cpu().numpy()

            for i in range(self.n_lines):
                a = an[0, i, :] * sc
                b = bn[0, i, :] * sc
                c = cn[0, i, :] * sc
                gray = int(np.clip(255 * (1 - self.lc[0, i, 0].item()) + 0.5, 0, 255))
                lw = sc * self.lw[0, i, 0].item()

                svg_data += f'  <path d="M{a[1]},{a[0]} Q{b[1]},{b[0]} {c[1]},{c[0]}" stroke="rgb({gray}, {gray}, {gray})" stroke-width="{lw}" fill="transparent"/>\n'

            svg_data += '</svg>'

            with open(svg_path, 'w') as svg_file:
                svg_file.write(svg_data)

    def load_from_svg(self, svg_path):
        tree = ET.parse(svg_path)
        root = tree.getroot()
        width = float(root.attrib['width'])
        height = float(root.attrib['height'])
        sc = min(width, height)

        paths = root.findall('{http://www.w3.org/2000/svg}path')
        n_lines = len(paths)

        a_arr = np.zeros((1, n_lines, 2), np.float32)
        b_arr = np.zeros((1, n_lines, 2), np.float32)
        c_arr = np.zeros((1, n_lines, 2), np.float32)
        lc_arr = np.zeros((1, n_lines, 1), np.float32)
        lw_arr = np.zeros((1, n_lines, 1), np.float32)

        for i, path in enumerate(paths):
            # Expected format: "M{x1},{y1} Q{x2},{y2} {x3},{y3}"
            parts = path.attrib['d'].split()
            # Parse starting point (stored as M{a[1]},{a[0]})
            m_coords = list(map(float, parts[0][1:].split(',')))
            # Parse control point (stored as Q{b[1]},{b[0]})
            q_coords = list(map(float, parts[1][1:].split(',')))
            # Parse end point (stored as {c[1]},{c[0]})
            c_coords = list(map(float, parts[2].split(',')))

            # Reverse the coordinate swap and scaling (original = [y, x] / sc)
            a_arr[0, i, :] = [m_coords[1] / sc, m_coords[0] / sc]
            b_arr[0, i, :] = [q_coords[1] / sc, q_coords[0] / sc]
            c_arr[0, i, :] = [c_coords[1] / sc, c_coords[0] / sc]

            # Parse stroke: "rgb(gray, gray, gray)"
            inside = path.attrib['stroke'][4:-1]
            gray = int(inside.split(',')[0])
            lc_arr[0, i, 0] = 1 - gray / 255.0

            # Const init line width
            lw_arr[0, i, 0] = (1.0 / min(*self.img_shape)) / 2

        self.a = nn.Parameter(torch.tensor(a_arr, device=self.a.device))
        self.b = nn.Parameter(torch.tensor(b_arr, device=self.a.device))
        self.c = nn.Parameter(torch.tensor(c_arr, device=self.a.device))
        self.lc = nn.Parameter(torch.tensor(lc_arr, device=self.a.device))
        self.lw = nn.Parameter(torch.tensor(lw_arr, device=self.a.device))
        self.n_lines = n_lines

    def regularize_len(self):
        """Punish long splines."""

        # Approximate length
        ab = self.a[0] - self.b[0]
        bc = self.b[0] - self.c[0]
        lengths = (torch.norm(ab, dim=1) + torch.norm(bc, dim=1))
        return torch.sum(torch.square(lengths)) * (1.0 / 64.0)

    def forward(self):
        # Return parameters directly
        return self.a, self.b, self.c, self.lw, self.lc


class QuadraticSplineRenderer(nn.Module):
    def __init__(self, img_shape=(512, 512), tile_size=(64, 64), margin=0, quantize=None):
        """
        A self-contained stroke renderer for quadratic splines.

        This does two things to save on memory:
        1) Tile rendering: We carve up the image into smaller tiles; for each tile, we only render splines with bounding
           boxes that overlap the tile.
        2) Spline chunking: Within each tile, we consider only a subset of splines at a time.

        TODO: Sort splines - best if there are fewer chunks per tile

        :param img_shape: Shape of rendered image, (h, w)
        :param tile_size: Size of tiles for rendering, (tile_h, tile_w)
        margin: extra margin to expand each spline's bounding box
        """
        super().__init__()
        self.img_shape = img_shape
        self.tile_size = tile_size
        self.margin = margin
        self.quantize = quantize

        # Texture coordinates
        # Register these as a buffer so .to(device) works
        yy, xx = torch.meshgrid(torch.linspace(0.0, self.img_shape[0] / min(*self.img_shape), self.img_shape[0]),
                                torch.linspace(0.0, self.img_shape[1] / min(*self.img_shape), self.img_shape[1]),
                                indexing='ij')
        self.register_buffer("p", torch.stack([yy, xx], dim=-1), persistent=False)

    def solve_cubic(self, ax, ay, az):
        """
        Solve cubic t^3 + ax t^2 + ay t + az = 0
        Return shape (..., 2) with the relevant two roots we keep.
        """

        # cache some reused values
        ax_2 = ax * ax
        p = ay - (ax_2 / 3.0)
        p3 = p * p * p
        q = ax * (2.0 * ax_2 - 9.0 * ay) / 27.0 + az

        d = q * q + 4.0 * p3 / 27.0

        # case d > 0: one root
        sqrt_d = safe_sqrt(d)
        x0 = cubrt((sqrt_d - q) * 0.5)
        x1 = cubrt((-sqrt_d - q) * 0.5)

        root0 = x0 + x1 - ax / 3.0

        # case d <= 0: technically 3 roots, but the center one can't be the closest so we'll ignore it.
        p3_sdiv = torch.where(abs(p3) > 1e-9, p3, torch.ones_like(p3) * 1e-9)

        v = safe_acos(-safe_sqrt(-27.0 / p3_sdiv) * q * 0.5) / 3.0
        m = torch.cos(v)
        n = torch.sin(v) * np.sqrt(3)

        root1 = (m + m) * safe_sqrt(-p / 3.) - ax / 3.
        root2 = (-n - m) * safe_sqrt(-p / 3.) - ax / 3.

        roots_single = torch.cat([root0, root0], dim=-1)  # shape (...,2)
        roots_double = torch.cat([root1, root2], dim=-1)  # shape (...,2)

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

        # This is a simple guard against point b being at (0, 0), which would make this blow up.
        b = b + 1e-5

        tile_h, tile_w, _ = p.shape
        p_flat = p.reshape(-1, 2)  # (tile_h*tile_w, 2). Can't be a view, the tile p is a non-contiguous slice

        # Reshapes for broadcasting
        p_flat = p_flat.unsqueeze(0).unsqueeze(0)  # shape (1,1,tile_h*tile_w,2)
        A = (b - a)
        B_ = (c - b) - A
        C = p_flat - a.unsqueeze(2)
        D = 2.0 * A.unsqueeze(2)

        B_dot_B = torch.sum(B_.unsqueeze(2) * B_.unsqueeze(2), dim=-1, keepdim=True)

        cubic_a = (-3.0 * torch.sum(A.unsqueeze(2) * B_.unsqueeze(2), dim=-1, keepdim=True)) / (-B_dot_B)
        cubic_b = ((torch.sum(C * B_.unsqueeze(2), dim=-1, keepdim=True) - 2.0 * torch.sum(A.unsqueeze(2) * A.unsqueeze(2), dim=-1, keepdim=True)) / (-B_dot_B))
        cubic_c = (torch.sum(C * A.unsqueeze(2), dim=-1, keepdim=True)) / (-B_dot_B)

        roots = self.solve_cubic(cubic_a, cubic_b, cubic_c)  # (B, M, tile_h*tile_w, 2)

        # T = roots
        T = torch.clamp(roots, 0.0, 1.0)

        d1 = (D + B_.unsqueeze(2) * T[..., 0:1]) * T[..., 0:1] - C
        d2 = (D + B_.unsqueeze(2) * T[..., 1:2]) * T[..., 1:2] - C

        dd1 = torch.sum(d1 * d1, dim=-1)
        dd2 = torch.sum(d2 * d2, dim=-1)
        dist = safe_sqrt(torch.minimum(dd1, dd2))

        # Reshape to (B,M,tile_h,tile_w,1)
        dist = dist.view(dist.shape[0], dist.shape[1], tile_h, tile_w, 1)
        return dist

    @torch.no_grad()
    def _compute_spline_bbox_px(self, a, b, c, margin=2):
        """
        Return (y0,y1, x0,x1) in pixel coords for one spline a,b,c in [0..1].
        """

        # scale factor to pixel space
        SC = np.array([min(*self.img_shape)] * 2)

        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()

        # endpoint bbox
        mi = np.minimum(a, c)
        ma = np.maximum(a, c)

        # if control point b is outside the (a, c) bbox
        if b[0] < mi[0] or b[0] > ma[0] or b[1] < mi[1] or b[1] > ma[1]:
            denom = (a - 2.0 * b + c)
            if (np.abs(denom) > 1e-9).all():
                t = np.clip((a - b) / (a - 2.0 * b + c), a_min=0.0, a_max=1.0)
                s = 1.0 - t
                q = s * s * a + 2.0 * s * t * b + t * t * c

                mi = np.minimum(mi, q)
                ma = np.maximum(ma, q)

        mi = np.clip(np.floor(mi * SC) - margin, 0, self.img_shape).astype(int)
        ma = np.clip(np.ceil(ma * SC) + margin, 0, self.img_shape).astype(int)

        return mi[0], ma[0], mi[1], ma[1]

    def forward(self, a, b, c, lw, lc, chunk_size_splines=384):  # 384
        """
        a,b,c: (B, N, 2) control points in [0,1].
        lw, lc: (B, N) line widths, line color
        chunk_size_splines: how many splines we process at once if there's a large # of them in a tile.

        Return: (B,1,H,W)
        """
        B, N, _ = a.shape
        H, W = self.img_shape
        tile_h, tile_w = self.tile_size
        device = a.device
        dtype = a.dtype

        # Optional quantization of line color via the detach trick
        if self.quantize:
            scale = self.quantize - 1
            lcq = torch.round(lc * scale ) / scale
            lc = lc + (lcq - lc ).detach()

        # Compute AABBs for all splines
        bboxes = []
        with torch.no_grad():
            for bi in range(B):
                row_bboxes = []
                for ni in range(N):
                    bb = self._compute_spline_bbox_px(a[bi, ni], b[bi, ni], c[bi, ni], margin=int(self.margin + 3*lw[bi, ni]))
                    row_bboxes.append(bb)
                bboxes.append(row_bboxes)

        out = torch.zeros((B, 1, H, W), dtype=dtype, device=device)

        for ty in range(0, H, tile_h):
            for tx in range(0, W, tile_w):
                tile_y1 = min(ty + tile_h, H)
                tile_x1 = min(tx + tile_w, W)

                # tile area => [ty:tile_y1, tx:tile_x1]

                tile_height = tile_y1 - ty
                tile_width = tile_x1 - tx
                if tile_height <= 0 or tile_width <= 0:
                    continue

                # Figure out which splines intersect this tile.
                # Address by batch: tile_splines[b] = [indices of splines this batch that intersect this tile]
                tile_splines = [[] for _ in range(B)]
                for bi in range(B):
                    for ni in range(N):
                        y0, y1, x0, x1 = bboxes[bi][ni]
                        if y1 > ty and y0 < tile_y1 and x1 > tx and x0 < tile_x1:
                            tile_splines[bi].append(ni)

                # Empty tile?
                if all(len(slist) == 0 for slist in tile_splines):
                    continue

                # Slice out tile indices
                p_sub = self.p[ty:tile_y1, tx:tile_x1].to(device)

                tile_out = torch.zeros((B, tile_height, tile_width), dtype=dtype, device=device)

                for bi in range(B):
                    # spline indices that touch this tile, this batch
                    sids = tile_splines[bi]
                    if len(sids) == 0:
                        continue

                    for start in range(0, len(sids), chunk_size_splines):
                        end = min(start + chunk_size_splines, len(sids))
                        these_sids = sids[start:end]  # list of spline idx

                        a_ = a[bi, these_sids, :].unsqueeze(0)  # shape (1, M_chunk,2)
                        b_ = b[bi, these_sids, :].unsqueeze(0)
                        c_ = c[bi, these_sids, :].unsqueeze(0)
                        lw_ = lw[bi, these_sids]  # shape (M_chunk,)
                        lc_ = lc[bi, these_sids]

                        # compute distance => (1, M_chunk, tile_h, tile_w, 1)
                        dist = self.quadratic_bezier_distance(p_sub, a_, b_, c_)

                        # Compute intensity
                        dist2d = dist[0, :, :, :, 0]
                        lw_2d = lw_.view(-1, 1, 1)
                        lc_2d = lc_.view(-1, 1, 1)
                        intensity = torch.sigmoid((lw_2d - dist2d) / lw_2d * 6.0) * lc_2d

                        # Accumulate this tile
                        tile_out[bi] += intensity.sum(dim=0)

                # Accumulate tiles into composite image
                out[:, 0, ty:tile_y1, tx:tile_x1] += tile_out

        return 1.0 - out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchinfo
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (123, 456)

    line_params = QuadraticSplineParams(n_lines=600, img_shape=img_shape).to(device)
    line_params.init_lines()

    lines = QuadraticSplineRenderer(img_shape=img_shape).to(device)

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
