# semantic-plot
Generate human-like line art for plotters from images, via differentiable rendering and perceptual losses

## Getting Started

### Dependencies / Setup
Install pytorch and torchvision from: https://pytorch.org.<br>
Be sure to install a version with GPU/CUDA support. This project works best on GPUs with at least 8GB of VRAM. 

Clone / Install Dependencies: 
```shell
git clone https://github.com/evanfletcher42/semantic-plot
cd semantic-plot
pip install -r requirements.txt
```

## How to Use (The Short Version)
This project works in two steps:
1. **Solve for Splines**: Solve for a set of (grayscale) splines for a target image.
2. **Post-Process**: Turn these grayscale splines into discrete strokes for a plotter.that

### Step 1: Solve for Splines
Optimize a set of splines for an image, from scratch:
```shell
$ python train_spline_img_lpips.py -i path/to/input.png 
```

This will run forever, and will save SVGs + preview images under ./outputs/ when new best-candidates are found.<br />
**Stop (CTRL + C) when things look good enough.**  

For full options, including changing the number of splines or the size of the image target, see `python train_spline_img_lpips.py -h`.

With default settings on a RTX 2080ti, most pictures:
 - start looking pretty good after 250 iterations (~10 mins)
 - look great after ~500 iterations (~20 mins)
 - Will continue to show minor-but-visible improvement until ~1500 iterations or so

... but this will depend on your image, settings, hardware, and patience. 

The SVG files produced by this step will contain splines with arbitrary gray levels. Most plotters can't draw arbitrary 
gray levels, so we need to post-process this into something a plotter can draw: stacked strokes.  

### Step 2: Post-Processing for Plotters

```shell
python postprocess_svg.py -i path/to/your/best_solved.svg -o output.svg --skip-halflevel
```

This will turn grayscale splines into repeated strokes, based on line intensity. It will also add a bit of jitter to make the drawn curves seem less mechanical.

For full options, see `python postprocess_svg.py -h` or "Advanced Usage" below.

## Advanced Usage and Tips

### Quantized Spline Refinement
By default, `train_spline_img_lpips.py` will solve for a set of splines from scratch with arbitrary floating-point intensity.
This is good for the solver, but is generally not how plotters work. Most plotters can only change the apparent weight
of a line by drawing several lines on top of, or near, each other. We can mimic this effect by quantizing spline
intensities.

To quantize an existing drawing to a particular maximum number of overlapping lines, then solve with this constraint:
```shell
python train_spline_img_lpips.py -i path/to/input.png -q 13 --init-svg path/to/previous_solve.svg
```
This will:
 - Specify that the output will be constrained to 13 gray levels (up to 12 overlapping strokes + one slot for "no stroke")
 - Initialize using a previous solved SVG file

but otherwise proceed normally. When done, run `postprocess_svg.py` with `--gray-levels` set to the same number as used here.

Quantized solves are best used for refinement, after a regular solve is done. 

### Managing GPU Memory
During a solve (`train_spline_img_lpips.py`), current memory usage is printed to stdout per iteration. The number will
vary over time, as the solver's tiling/batching interacts with splines as they move around.

Required memory mostly depends on image size (`-s`,`--size-min`) and number of drawn splines (`-n`,`--num-splines`).
If you find yourself running out of memory during a solve, or see the process slow to a crawl as your system starts 
using shared GPU memory, consider tweaking these options.



### Common Post-Processing Tweaks

The default settings in `postprocess_svg.py` work decently well for the author's plotter, pens, and paper. Common
options to tweak include:
 - `--gray-levels`: Number of discrete gray levels to represent with stacked strokes. Default 12 (= max 11 stacked lines, plus "no line").
 - `--skip-halflevel`: If set, this script will output fewer faint lines, not drawing any under 1/2 of a gray level (as 
set by `--gray-levels`). Try toggling this if your drawing has many faint lines. 
 - `--wobble`: Adds variations to strokes to make them look more hand-drawn, instead of mechanical and perfect splines.


## Full Options

#### train_spline_img_lpips.py -h:
```
usage: train_spline_img_lpips.py [-h] -i TARGET_PATH [-o OUTPUT_PATH] [-n NUM_SPLINES] [-s SIZE_MIN] [-q QUANTIZE]
                                 [--init-svg INIT_SVG]

Computes a set of splines that semantically match an input image

options:
  -h, --help            show this help message and exit
  -i TARGET_PATH, --target-path TARGET_PATH
                        Target image path (required). (default: None)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Outputs directory. Each run will create a folder in this directory. Will be created if missing. (default:
                        outputs)
  -n NUM_SPLINES, --num-splines NUM_SPLINES
                        Number of splines to draw. (default: 3600)
  -s SIZE_MIN, --size-min SIZE_MIN
                        Minimum dimension of the rendered image. Target images will be resized to this. (default: 384)
  -q QUANTIZE, --quantize QUANTIZE
                        Optional quantization of line weight to a specified number of overlaid strokes. No quantization if
                        unspecified. (default: None)
  --reg-sl REG_SL       Weighting factor for spline-length regularization. If zero, not computed. (default: 0.0)
  --init-svg INIT_SVG   Optional spline init from a SVG file from a previous run, instead of from scratch. Note: Overrides some
                        settings, like number of lines or image size. (default: None)
```

#### postprocess_svg.py -h:

```
postprocess_svg.py [-h] --input INPUT --output OUTPUT [--gray-levels GRAY_LEVELS] [--jitter-std JITTER_STD]
                          [--min-threshold MIN_THRESHOLD] [--skip-halflevel] [--sort-right] [--wobble]

Postprocess solved SVG for plotter output

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to input SVG file
  --output OUTPUT, -o OUTPUT
                        Path to output SVG file
  --gray-levels GRAY_LEVELS
                        Number of gray levels
  --jitter-std JITTER_STD
                        Standard deviation for jitter
  --min-threshold MIN_THRESHOLD
                        Minimum brightness threshold
  --skip-halflevel      Skip very light lines
  --sort-right          Make the first stroke in a set move from left to right (handy for certain pens)
  --wobble              Enable hand wobble on first stroke
```
