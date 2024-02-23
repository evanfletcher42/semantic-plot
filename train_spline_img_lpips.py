import torch
from ttools.modules.losses import LPIPS
import cv2
import matplotlib.pyplot as plt
from spline_render import QuadraticSplineRenderer
import os
import time
from pathlib import Path
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "data/alyx.png"
draw_sz = (256, 256)

target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
target_img = cv2.resize(target_img, draw_sz, interpolation=cv2.INTER_AREA)

plt.imshow(target_img, cmap='gray')
plt.show()

target = torch.tensor(target_img / 255.0, dtype=torch.float32).to(device)

lines = QuadraticSplineRenderer(n_lines=768, img_shape=draw_sz, init_img=target_img).to(device)

optim = torch.optim.Adam(
    [
        {"params": lines.a, "lr": 0.005},
        {"params": lines.b, "lr": 0.005},
        {"params": lines.c, "lr": 0.005},
        # {"params": lines.lw, "lr": 0.001},
        {"params": lines.lc, "lr": 0.004}
    ]
)

tstr = time.strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join("outputs", Path(os.path.basename(img_path)).stem + "_" + tstr)
os.makedirs(out_dir, exist_ok=False)

perceptual_loss = LPIPS().to(device)

for i in range(1000000):
    start_t = time.perf_counter()

    lines.zero_grad()
    img_render = lines()

    err = perceptual_loss(img_render[None, ...], target[None, ...])

    err.backward()
    optim.step()

    # sanity clamping
    with torch.no_grad():
        lines.a.clamp_(0.0, 1.0)
        lines.b.clamp_(0.0, 1.0)
        lines.c.clamp_(0.0, 1.0)
        lines.lw.clamp_(1e-5, 0.10)
        lines.lc.clamp_(0.0, 1.0)

    img_np = np.clip(img_render.detach().cpu().numpy() * 255, 0, 255)
    cv2.imwrite(os.path.join(out_dir, "%06d.png" % i), img_np[0, ...])

    end_t = time.perf_counter()
    print("Iter %d Loss %f Time %f" % (i, err.item(), end_t-start_t))

