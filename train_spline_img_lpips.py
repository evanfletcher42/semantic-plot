import torch
from ttools.modules.losses import LPIPS
import cv2
import matplotlib.pyplot as plt
from spline_render import QuadraticSplineParams, QuadraticSplineRenderer
import os
import time
from pathlib import Path
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "data/alyx.png"
    draw_sz = (256, 256)
    n_lines = 768

    target_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.resize(target_img, draw_sz, interpolation=cv2.INTER_AREA)

    plt.imshow(target_img, cmap='gray')
    plt.show()

    target = torch.tensor(target_img / 255.0, dtype=torch.float32).to(device)

    line_params = QuadraticSplineParams(n_lines=n_lines, img_shape=draw_sz).to(device)
    line_params.init_lines(target_img)

    lines = QuadraticSplineRenderer(img_shape=draw_sz).to(device)

    optim = torch.optim.Adam(
        [
            {"params": line_params.a, "lr": 0.005},
            {"params": line_params.b, "lr": 0.005},
            {"params": line_params.c, "lr": 0.005},
            # {"params": line_params.lw, "lr": 0.001},
            {"params": line_params.lc, "lr": 0.004}
        ]
    )

    tstr = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("outputs", Path(os.path.basename(img_path)).stem + "_" + tstr)
    os.makedirs(out_dir, exist_ok=False)

    perceptual_loss = LPIPS().to(device)

    for i in range(1000000):
        start_t = time.perf_counter()

        line_params.zero_grad()
        img_render = lines(*line_params())

        err = perceptual_loss(img_render, target[None, ...])

        err.backward()
        optim.step()

        # sanity clamping
        with torch.no_grad():
            line_params.a.clamp_(0.0, 1.0)
            line_params.b.clamp_(0.0, 1.0)
            line_params.c.clamp_(0.0, 1.0)
            line_params.lw.clamp_(1e-5, 0.10)
            line_params.lc.clamp_(0.0, 1.0)

        img_np = np.clip(img_render.detach().cpu().numpy() * 255, 0, 255)
        cv2.imwrite(os.path.join(out_dir, "%06d.png" % i), img_np[0, 0, ...])

        end_t = time.perf_counter()
        print("Iter %d Loss %f Time %f" % (i, err.item(), end_t-start_t))


if __name__ == "__main__":
    main()
