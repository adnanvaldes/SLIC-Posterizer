# SLICPosterizer

`SLICPosterizer` is a Python tool for artistic image posterization that transforms photographs into stylized artwork with reduced color palettes. It provides both a command-line interface and Python API for fine control over colors, edges, blur, and segmentation.

The program is inspired by [PosterChild](https://github.com/tedchao/PosterChild), although this implementation is my own. The primary difference
is that in `PosterChild` the authors use `KMeans` from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) as well
as [SciPy ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html). While their approach offers better visual results, I found
the use of these algorithms (in particular KMeans) very slow on my computer. 

The result is a faster posterization engine based entirely on [SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/) superpixels. It works because
SLIC itself is also a clustering algorithm, like KMeans, but optimized for image segmentation. Instead of clustering pixels based on color similarity, SLIC
creates clusters based both on color similarity _and_ spatial proximity.

This implementation also streamlines dependencies and tries to use only the necessary imports to function (without spending too long trying to re-implement base algorithms and functionality).

---

## Features

- SLIC superpixel segmentation with adjustable compactness and count  
- Color palette optimization and posterization  
- Edge preservation with threshold controls  
- Optional overlay of superpixel boundaries  
- Command-line usage or import as a Python module  

---

## Installation

You can use the package either:

- Directly by running the script file, or  
- By building and installing it locally via [uv](https://uv.run/):

```bash
uv build
uv pip install -e .
```

---

## Usage

### CLI

Run the CLI tool with:

```bash
slicposterizer input.jpg output.jpg [options]
```

```bash
usage: slicposterizer [-h] [-m MIXING] [-p PALETTE] [-c COLORS] [-S SUPERPIXELS]
                      [--compactness COMPACTNESS] [-b BLUR] [-e EDGE_THRESHOLD]
                      [-d DOWNSAMPLE] [--no-edge-preserve] [--detail-blend DETAIL_BLEND]
                      [--quality QUALITY] [-s [1-10]] [--overlay]
                      input output

SLIC Posterizer - SLIC-Based Artistic Posterization

positional arguments:
  input                 Input image path
  output                Output posterized image path

options:
  -h, --help            show this help message and exit
  -m MIXING, --mixing MIXING
                        Output prefix for additive mixing layers (default: None)
  -p PALETTE, --palette PALETTE
                        Output path for palette swatch (default: None)
  -c COLORS, --colors COLORS
                        Number of colors in palette (default: 64)
  -S SUPERPIXELS, --superpixels SUPERPIXELS
                        Number of superpixels (default: 4500)
  --compactness COMPACTNESS
                        SLIC superpixel compactness parameter (default: 15.0)
  -b BLUR, --blur BLUR  Blur radius for smoothing (default: 1)
  -e EDGE_THRESHOLD, --edge-threshold EDGE_THRESHOLD
                        Edge detection threshold (default: 0.1)
  -d DOWNSAMPLE, --downsample DOWNSAMPLE
                        Downsample factor (>=1) (default: 1)
  --no-edge-preserve    Disable edge preservation (default: False)
  --detail-blend DETAIL_BLEND
                        Blend factor for detail preservation (default: 0.1)
  --quality QUALITY     JPEG quality (if saving JPEG) (default: 95)
  -s [1-10], --smoothing [1-10]
                        Smoothing strength level (1-10) (default: 3)
  --overlay             Overlay superpixel boundaries on the final image (default: False)
  ```
---

### Python API

Use the package programmatically by importing `posterize`:

```python
from slicposterizer import posterize

posterize(
    input_path="input.jpg",
    output_path="output.jpg",
    num_colors=64,
    num_superpixels=4500,
    blur_radius=1.0,
    edge_threshold=0.1,
    preserve_edges=True,
    overlay_superpixels=False,
)
```

---

## License

This project is licensed under the GPL-3.0-or-later license. See the [LICENSE](LICENSE) file for details.

---

## Author

Adnan Valdes - [blog](https://arvb.net)

---

## Acknowledgments

Chao, C.-K. T., Singh, K., & Gingold, Y. (2021). PosterChild: Blend-Aware Artistic Posterization. Computer Graphics Forum, 40(4), 87–99. https://doi.org/10.1111/cgf.14343
scholar.google.com.co+4ResearchGate+4Google Scholar+4
[PosterChild: Blend-Aware Artistic Posterization (PDF)](https://cragl.cs.gmu.edu/posterchild/PosterChild-%20Blend-Aware%20Artistic%20Posterization%20(Cheng-Kang%20Ted%20Chao,%20Karan%20Singh,%20Yotam%20Gingold%202021%20EGSR)%20300dpi.pdf)

Achanta, R., Shaji, A., & Smith, K. (2010). SLIC Superpixels. EPFL Report, 2. https://infoscience.epfl.ch/entities/publication/2dd26d47-3d00-43eb-9e31-4610db94a26e
Infoscience
[SLIC Superpixels (PDF)](https://infoscience.epfl.ch/entities/publication/2dd26d47-3d00-43eb-9e31-4610db94a26e)