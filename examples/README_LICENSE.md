## Attribution and Licensing 
Images used in these examples are all from Wikimedia Commons, and were marked for reuse under Creative Commons terms.
The specific license for each image is noted below. Some images were slightly cropped prior to processing. 

Plotted images are under the same Creative Commons terms as their source image. 

### wroclaw_collegium_maximum.jpg
[Source Link](https://commons.wikimedia.org/wiki/Template:Potd/2025-03#/media/File:Wroclaw_-_wieza_Uniwersytetu_Wroclawskiego_z_ksiezycem_w_tle.jpg)<br/>
Photo by [Jar.Ciurus on Wikimedia Commons](https://commons.wikimedia.org/wiki/User:Jar.ciurus)<br/>
[CC BY-SA 3.0 pl](https://creativecommons.org/licenses/by-sa/3.0/pl/)

### catrin.jpg
[Source Link](https://commons.wikimedia.org/wiki/Category:Facepainting#/media/File:Catr%C3%ADn_en_Chapala.jpg)<br/>
Photo by [Wendymtz06 on Wikimedia Commons](https://commons.wikimedia.org/w/index.php?title=User:Wendymtz06)<br/>
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

### pride_of_baltimore_II.jpg
[Source Link](https://commons.wikimedia.org/wiki/Template:Potd/2011-10#/media/File:NS_-_Pride_of_Baltimore_II.jpg)<br/>
Photo by [Taxiarchos228 on Wikimedia Commons](https://de.wikipedia.org/wiki/Benutzer_Diskussion:Taxiarchos228)<br/>
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

### Hirundo_rustica_Ormoz.jpg
[Source Link](https://commons.wikimedia.org/wiki/Commons:Picture_of_the_day#/media/File:Hirundo_rustica_Ormoz.jpg)<br/>
Photo by [Yerpo on Wikimedia Commons](https://commons.wikimedia.org/wiki/User:Yerpo)<br/>
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Recreating These Plots
Plotted SVG files were generated with the following commands:

```shell
# Initial solve, run for ~1000 iterations
python3 train_spline_img_lpips.py -i <image>

# Quantized refinement, run for ~500 iterations
python3 train_spline_img_lipis.py -i <image> --init-svg <last_step_output.svg> --quantize 15

# Post-processing
python3 postprocess_svg.py -i <quantized_output.svg> -o <postprocessed.svg> --gray-levels 15 --skip-halflevel
```

Results were plotted on an iDraw H pen plotter on cardstock, using a LAMY Safari fountain pen, a refillable cartridge, 
and Pilot Iroshizuku tsuki-yo ink.
