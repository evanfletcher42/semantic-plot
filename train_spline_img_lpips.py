import torch
import cv2
import matplotlib.pyplot as plt
from spline_render import QuadraticSplineParams, QuadraticSplineRenderer
import os
import time
from pathlib import Path
import numpy as np
from semantic_loss import CachedLPIPS


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_path = "data/eagle-3.png"
    draw_sz_min = 384  # Smallest dimension of input image, when resized

    n_lines = 3600

    quantize_lc_n = None  # No quantization (use for full solve from scratch)
    # quantize_lc_n = 16 + 1  # Quantization levels for refinement, including "no line"

    if quantize_lc_n is None:
        warmup_n = 125  # Don't save images every iteration up to this many iterations, to speed up early stages
    else:
        warmup_n = 0

    if quantize_lc_n is None:
        start_reset_n = 10  # Start resetting lines after this many iterations without improvement
        stop_reset_n = 50  # If we don't have a new best loss after this many iterations, stop resetting invisible lines
    else:
        start_reset_n = 100
        stop_reset_n = 110

    settle_n = 300  # Terminate after this many additional steps without a new best past stop_reset_n

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize
    if img is None:
        raise ValueError(f"Cannot load image at {img_path}")
    h, w, _ = img.shape
    scale = draw_sz_min / min(h, w)
    new_size = (int(w * scale), int(h * scale))
    target_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    draw_sz = target_img.shape[:2]

    # parameter clamp factors: Smaller image dimension is 1.0 in spline space
    clamp_val = (draw_sz[0] / min(*draw_sz), draw_sz[1]/min(*draw_sz))
    clamp_t = torch.tensor(clamp_val, device=device)

    plt.imshow(target_img, cmap='gray')
    plt.show()

    # to channels first
    target_img = np.moveaxis(target_img, -1, 0)

    target = torch.tensor(target_img / 255.0, dtype=torch.float32).to(device)
    target_gray = torch.tensor(target_img_gray / 255.0, dtype=torch.float32).to(device)
    perceptual_loss = CachedLPIPS().to(device)
    perceptual_loss.set_target(target, target_gray)

    lines = QuadraticSplineRenderer(img_shape=draw_sz, quantize=quantize_lc_n).to(device)

    line_params = QuadraticSplineParams(n_lines=n_lines, img_shape=draw_sz).to(device)

    # Init from empty
    line_params.init_lines(renderer=lines, loss_func=perceptual_loss)

    # Init from file (for refinement solve)
    # line_params.load_from_svg("path/to/strokes.svg")
    if quantize_lc_n is not None:
        line_params.min_intensity = 1.0 / (quantize_lc_n - 1) / 2

    if quantize_lc_n is None:
        optim = torch.optim.Adam(
            [
                {"params": line_params.a, "lr": 0.00375},
                {"params": line_params.b, "lr": 0.00375},
                {"params": line_params.c, "lr": 0.00375},
                # {"params": line_params.lw, "lr": 0.001},
                {"params": line_params.lc, "lr": 0.00300}
            ],
            amsgrad=True
        )
    else:
        # lower learning rates for refinement solves
        optim = torch.optim.Adam(
            [
                {"params": line_params.a, "lr": 0.00125},
                {"params": line_params.b, "lr": 0.00125},
                {"params": line_params.c, "lr": 0.00125},
                # {"params": line_params.lw, "lr": 0.001},
                {"params": line_params.lc, "lr": 0.00100}
            ],
            amsgrad=True
        )

    tstr = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("outputs", Path(os.path.basename(img_path)).stem + "_" + tstr)
    svg_dir = os.path.join(out_dir, "svg")
    os.makedirs(svg_dir, exist_ok=False)

    best_loss = 1e9

    steps_since_best = 0

    for i in range(1000000):
        start_t = time.perf_counter()

        line_params.zero_grad()
        img_render = lines(*line_params())
        err_perceptual = perceptual_loss(img_render)

        if quantize_lc_n is not None:
            # regularize lines for not-scribblyness
            err_reg_len = line_params.regularize_len()
            err = err_perceptual + err_reg_len
        else:
            err = err_perceptual
            err_reg_len = torch.Tensor([0])

        err.backward()

        optim.step()

        # sanity clamping
        with torch.no_grad():
            line_params.a.clamp_(torch.zeros_like(clamp_t), clamp_t)
            line_params.b.clamp_(torch.zeros_like(clamp_t), clamp_t)
            line_params.c.clamp_(torch.zeros_like(clamp_t), clamp_t)
            line_params.lw.clamp_(1e-5, 0.10)
            line_params.lc.clamp_(0.0, 1.0)

        loss = err.item()

        if loss < best_loss:
            best_loss = loss
            steps_since_best = 0
        else:
            steps_since_best += 1

        if ( i < warmup_n and best_loss == loss and i % 10 == 0) or (best_loss == loss and i >= warmup_n) or ( i % 50 == 0 ):
            img_np = np.clip(img_render.detach().cpu().numpy() * 255, 0, 255)
            cv2.imwrite(os.path.join(out_dir, "%06d_%0.04f_%0.04f.png" % (i, loss, best_loss)), img_np[0, 0, ...])
            line_params.save_svg(os.path.join(svg_dir, "%06d_%0.04f_%0.04f.svg" % (i, loss, best_loss)), img_sz=draw_sz)

        # reinit invisible if we are taking negative steps
        if start_reset_n <= steps_since_best < stop_reset_n:
            print("Reinit invisible...")
            line_params.reinit_invisible(curr_img=img_render, loss_func=perceptual_loss)

        end_t = time.perf_counter()
        print("Iter %d Loss %f Percep %f LReg %f Time %f Mem %f GB" % (i, loss, err_perceptual.item(), err_reg_len.item(), end_t-start_t, torch.cuda.max_memory_allocated(device)/1024/1024/1024) + (" ***" if loss == best_loss else ""))

        if steps_since_best >= stop_reset_n + settle_n:
            # Call it
            # print("Terminating after %d steps without improvement" % steps_since_best)
            # break

            # HACK: Force line reinitialization if we've been stuck here for a bit
            steps_since_best = 0

    print("Done")


if __name__ == "__main__":
    main()
    print("Exit")
