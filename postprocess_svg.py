"""
postprocess_svg.py

Applies postprocessing to a solved SVG, turning it into the actual paths you'd send to a plotter.
- Creates overlaid lines based on solved intensity, for representing lighter/darker edges
- Applies a bit of noise to output paths, to make them look a little less machine-drawn

Usage:
    python svg_postprocess.py \
        --input INPUT_SVG \
        --output OUTPUT_SVG \
        [--gray-levels 12] \
        [--jitter-std 0.35375] \
        [--min-threshold 0.0] \
        [--skip-halflevel] \
        [--sort-right] \
        [--wobble]
"""
import argparse
import re
import sys
from xml.etree import ElementTree as ET
import numpy as np
from tqdm import tqdm


def split_bezier(points, t):
    """Split a quadratic bezier via De Casteljau's method."""
    P0, P1, P2 = points
    P01 = (1 - t) * P0 + t * P1
    P12 = (1 - t) * P1 + t * P2
    P012 = (1 - t) * P01 + t * P12
    return (P0, P01, P012), (P012, P12, P2)


def eval_bezier(a, b, c, t):
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c


def len_bezier(a, b, c):
    """Approximate bezier length by discrete sampling."""
    t = np.linspace(0.0, 1.0, 32)[:, None]
    p = eval_bezier(a, b, c, t)
    dp = np.diff(p, axis=0)
    return np.sum(np.linalg.norm(dp, axis=-1))


def wobble_curve_svg(a, b, c, jitter_std):
    """Add hand wobble to a quadratic bezier for SVG paths."""
    curve_len = len_bezier(a, b, c)
    std = jitter_std * curve_len / 256
    dt = 0.1
    t = dt
    s0, _ = split_bezier((a, b, c), t)
    s0j = (s0[0], s0[1], s0[2] + np.random.normal(scale=std, size=2))
    chain = []
    while t < 1 - 1e-6:
        t = min(1.0, t + dt)
        p = eval_bezier(a, b, c, t)
        p += np.random.normal(scale=std, size=2)
        chain.append(p)
    chain[-1] = c
    d = f'M{s0j[0][0]},{s0j[0][1]} Q{s0j[1][0]},{s0j[1][1]} {s0j[2][0]},{s0j[2][1]} '
    for p in chain:
        d += f'T {p[0]},{p[1]} '
    return f'  <path d="{d}" stroke="black" stroke-width="0.25" fill="none"/>'


def process_svg(input_path, output_path, gray_levels, jitter_std, min_threshold, skip_halflevel, sort_right, wobble):
    # Parse SVG dimensions
    tree = ET.parse(input_path)
    root = tree.getroot()
    width = root.get('width')
    height = root.get('height')
    try:
        w = float(width)
        h = float(height)
    except Exception:
        print(f"Error: could not parse width/height (got '{width}', '{height}')", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r') as f:
        content = f.read()

    output_lines = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']

    pattern = r'M([\d.]+),([\d.]+) Q([\d.]+),([\d.]+) ([\d.]+),([\d.]+)" stroke="rgb\((\d+), \7, \7\)" stroke-width="([\d.]+)"'

    skipped = total = 0
    for a_, b_, c_, d_, e_, f_, g_, lw_ in tqdm(re.findall(pattern, content)):
        a = np.array([float(a_), float(b_)])
        bpt = np.array([float(c_), float(d_)])
        cpt = np.array([float(e_), float(f_)])
        intensity = int(g_)
        lw = float(lw_)

        dval = (255 - intensity) / 255
        n_mult_f = (dval - min_threshold) / (1 - min_threshold) * gray_levels
        n_mult = round(n_mult_f)
        resid = max(0.0, n_mult_f - n_mult)

        if n_mult == 0:
            if skip_halflevel:
                skipped += 1
                continue
            if n_mult_f > 0.1:
                n_mult = 1
            else:
                skipped += 1
                continue

        if sort_right:
            ab = np.linalg.norm(a - bpt)
            bc = np.linalg.norm(bpt - cpt)
            start = a if ab > bc else cpt
            if start[0] > bpt[0]:
                a, cpt = cpt, a

        for i in range(n_mult):
            if i == 0:
                ai, bi, ci = a, bpt, cpt
            else:
                ai = a + np.random.normal(scale=jitter_std, size=2)
                bi = bpt + np.random.normal(scale=jitter_std, size=2)
                ci = cpt + np.random.normal(scale=jitter_std, size=2)
            if i % 2:
                ai, ci = ci, ai

            if wobble:
                line = wobble_curve_svg(ai, bi, ci, jitter_std)
            else:
                line = (
                    f'  <path d="M{ai[0]},{ai[1]} Q{bi[0]},{bi[1]} {ci[0]},{ci[1]}" '
                    f'stroke="black" stroke-width="{lw}" fill="none"/>'
                )
            output_lines.append(line)
            total += 1

    output_lines.append('</svg>')

    with open(output_path, 'w') as out_f:
        out_f.write("\n".join(output_lines))

    print(f"Skipped {skipped} paths; drew {total} total.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Postprocess solved SVG for plotter output")
    parser.add_argument('--input', '-i', required=True, help='Path to input SVG file')
    parser.add_argument('--output', '-o', required=True, help='Path to output SVG file')
    parser.add_argument('--gray-levels', type=int, default=12, help='Number of gray levels')
    parser.add_argument('--jitter-std', type=float, default=0.7075 / 2, help='Standard deviation for jitter')
    parser.add_argument('--min-threshold', type=float, default=0.0, help='Minimum brightness threshold')
    parser.add_argument('--skip-halflevel', action='store_true', help='Skip very light lines')
    parser.add_argument('--sort-right', action='store_true', help='Make the first stroke in a set move from left to right (handy for certain pens)')
    parser.add_argument('--wobble', action='store_true', help='Enable hand wobble on first stroke')
    args = parser.parse_args()

    process_svg(
        args.input,
        args.output,
        args.gray_levels,
        args.jitter_std,
        args.min_threshold,
        args.skip_halflevel,
        args.sort_right,
        args.wobble
    )
