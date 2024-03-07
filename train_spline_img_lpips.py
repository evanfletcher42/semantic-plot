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

    img_path = "data/heeler_puppy.jpg"
    draw_sz = (256, 256)
    n_lines = 600

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
    svg_dir = os.path.join(out_dir, "svg")
    os.makedirs(svg_dir, exist_ok=False)

    perceptual_loss = LPIPS().to(device)

    best_loss = 1e9

    for i in range(1000000):
        start_t = time.perf_counter()

        line_params.zero_grad()
        img_render = lines(*line_params())

        # penalize big negative intensities
        # should allow some lines to sit atop one another, but will hurt large piles
        err_scribble = torch.sum(torch.square(torch.clamp(img_render, min=None, max=-3) + 3))

        # clip image
        img_render = torch.clip(img_render, min=0, max=1)

        err_perceptual = perceptual_loss(img_render, target[None, ...])

        err = err_scribble + err_perceptual
        err.backward()

        optim.step()

        # sanity clamping
        with torch.no_grad():
            line_params.a.clamp_(0.0, 1.0)
            line_params.b.clamp_(0.0, 1.0)
            line_params.c.clamp_(0.0, 1.0)
            line_params.lw.clamp_(1e-5, 0.10)
            line_params.lc.clamp_(0.0, 1.0)

        loss = err.item()

        if loss < best_loss:
            best_loss = loss
            img_np = np.clip(img_render.detach().cpu().numpy() * 255, 0, 255)
            cv2.imwrite(os.path.join(out_dir, "%06d_%0.04f.png" % (i, best_loss)), img_np[0, 0, ...])
            line_params.save_svg(os.path.join(svg_dir, "%06d_%0.04f.svg" % (i, best_loss)))
        else:
            # reinit invisible if we are taking negative steps
            line_params.reinit_invisible(init_img=target_img)

        end_t = time.perf_counter()
        print("Iter %d Loss %f (p %f sc %f) Time %f Mem %f GB" % (i, err.item(), err_perceptual.item(), err_scribble.item(), end_t-start_t, torch.cuda.max_memory_allocated(device)/1024/1024/1024) + (" ***" if loss == best_loss else ""))


if __name__ == "__main__":
    main()
