#!/usr/bin/env python3

"""
SLICPosterizer - SLIC-Based Artistic Posterization

Copyright (C) 2025 Adnan Valdes

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
import functools
import logging
import sys
import time

from io import BytesIO
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import nnls
from skimage import color as skcolor
from skimage.segmentation import slic


# Constants for edge detection thresholds scaling
CANNY_LOW_THRESHOLD_BASE = 50
CANNY_HIGH_THRESHOLD_BASE = 150
CANNY_THRESHOLD_MARGIN = 10
EDGE_DETECTION_SCALE = 1


# Min segment amount to cover image pixels, in percentage
COVERAGE_THRESHOLD = 0.95

# Gaussian blur kernel size formula multiplier and offset
BLUR_KERNEL_MULTIPLIER = 2
BLUR_KERNEL_OFFSET = 1

# Cartoon smoothing parameter bases
BILATERAL_DIAMETER_BASE = 5
BILATERAL_DIAMETER_STEP = 2
SIGMA_COLOR_BASE = 40
SIGMA_COLOR_STEP = 10
SIGMA_SPACE_BASE = 40
SIGMA_SPACE_STEP = 5
MEDIAN_BLUR_BASE = 3
MEDIAN_BLUR_STEP = 2


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_method(log_time: bool = False):
    """
    Decorator to log start and end of a method call.
    If log_time=True, it also logs duration.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger.info(f"Starting '{func.__name__}'...")
            start = time.time() if log_time else None
            result = func(self, *args, **kwargs)
            if log_time:
                duration = time.time() - start
                logger.info(f"Finished '{func.__name__}' in {duration:.2f} seconds.")
            else:
                logger.info(f"Finished '{func.__name__}'.")
            return result

        return wrapper

    return decorator


def log_step(msg_or_func):
    """
    Decorator factory to log a custom message before a method call.
    Used to mark intermediate processing steps inside bigger methods.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            msg = msg_or_func(self) if callable(msg_or_func) else msg_or_func
            logger.info(f"Starting: {msg}")
            result = func(self, *args, **kwargs)
            logger.info(f"Finished: {msg}")
            return result

        return wrapper

    return decorator


class SLICPosterizer:
    def __init__(
        self,
        num_colors: int = 64,
        blur_radius: float = 2.0,
        edge_threshold: float = 0.1,
        downsample_factor: int = 1,
        preserve_edges: bool = True,
        num_superpixels: int = 4000,
        superpixel_compactness: float = 15.0,
        detail_blend_strength: float = 0.1,
        smoothing: int = 5,
        overlay_superpixels: bool = False,
    ) -> None:
        if num_colors < 2:
            raise ValueError("Number of colors must be at least 2.")

        if not (1 <= smoothing <= 10):
            raise ValueError("smoothing must be between 1 and 10")

        if not (0.01 <= edge_threshold <= 1.0):
            raise ValueError("edge_threshold must be between 0.01 and 1.0")

        self.num_colors = num_colors
        self.blur_radius = blur_radius
        self.edge_threshold = edge_threshold
        self.downsample_factor = max(1, downsample_factor)
        self.preserve_edges = preserve_edges
        self.num_superpixels = num_superpixels
        self.superpixel_compactness = superpixel_compactness
        self.detail_blend_strength = detail_blend_strength
        self.smoothing = smoothing
        self.overlay_superpixels = overlay_superpixels

    @log_method(log_time=True)
    def posterize(
        self,
        input_path: Path | Image.Image | str | None,
        output_path: Path | str,
        palette_path: Path | str | None = None,
        mixing_prefix: Path | str | None = None,
        quality: int = 95,
    ) -> None:
        """
        This is the main user-facing method. It loads an input image, optionally downsamples it to speed up processing,
        applies the posterization pipeline (superpixels, palette extraction, detail preservation), upsamples back if needed,
        and saves the final posterized image.

        It also optionally saves a color palette swatch and additive color layers for further use.
        """
        image = self.load_image(input_path)
        image = self.downsample_image(image)

        posterized, weights, palette, segments = self.process(image)
        posterized = self.upsample_image(posterized, image.shape)

        if self.overlay_superpixels:
            # Upsample segments if needed
            if self.downsample_factor > 1:
                h, w = image.shape[:2]
                upsampled_segments = cv2.resize(
                    segments.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(segments.dtype)
            else:
                upsampled_segments = segments

            posterized = self._overlay_superpixel_boundaries(
                posterized, upsampled_segments
            )

        self.save_image(posterized, output_path, quality)

        if palette_path:
            swatch = self._create_palette_swatch(palette)
            self.save_image(swatch, palette_path)

        if mixing_prefix:
            self._save_additive_layers(weights, palette, mixing_prefix)

    @log_method(log_time=True)
    def process(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Posterization using SLIC superpixels

        This method does the heavy lifting. It preprocesses the image (like blurring), calculates superpixels,
        extracts a color palette from those segments, applies a basic color replacement, optionally preserves
        edges and fine details using superpixel weights and edge detection, smooths the image with bilateral
        and median filters, then quantizes the final image colors strictly to the palette.
        """
        img_rgb = self._preprocess(image)
        segments, palette_rgb, palette_lab = self._copmute_segments_and_palette(img_rgb)

        simple_post, img_lab = self._simple_posterize(img_rgb, palette_lab, palette_rgb)

        if self.preserve_edges:
            img_lab = skcolor.rgb2lab(img_rgb)
            simple_post = self._preserve_detail(
                img_rgb, img_lab, segments, palette_lab, palette_rgb, simple_post
            )

        posterized = self._smooth(simple_post)
        posterized, final_weights = self._final_quantize(
            posterized, palette_lab, palette_rgb
        )

        return posterized, final_weights, palette_rgb, segments

    # Image I/O and preprocessing
    def load_image(self, source: Path | Image.Image | str | None) -> np.ndarray:
        """
        Reads an image from the given file source, converting it into a standard
        RGB numpy array with 8-bit color channels.
        It raises errors if the file is missing or cannot be loaded properly,
        ensuring the rest of the pipeline starts with a valid image.

        Source parameter can be stdin, an Image, or a Path to an image file
        """

        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))

        if isinstance(source, str):
            source = Path(source)

        if source is None or source == Path("-"):
            if sys.stdin.isatty():
                raise ValueError("No input source provided and stdin is not piped.")
            source, source_name = sys.stdin.buffer, "stdin"
        else:

            if not source.exists():
                raise FileNotFoundError(f"File not found: {source}")
            source, source_name = source, str(source)

        try:
            with Image.open(source) as img:
                return np.array(img.convert("RGB"))
        except Exception as e:
            raise ValueError(f"Failed to load image from {source_name}: {e}") from e

    def save_image(self, img: np.ndarray, path: Path | str, quality: int = 95) -> None:
        """
        Converts the processed image, which uses floats between 0 and 1, back to 8-bit unsigned integers (0-255)
        and saves it to the specified path using Pillow.
        Quality parameter controls compression for formats like JPEG.
        """
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_u8).save(path, quality=quality)
        logger.info(f"Saved image: {path}")

    def downsample_image(self, img: np.ndarray) -> np.ndarray:
        """
        Reduces image dimensions by the downsample factor using OpenCV's area interpolation,
        which is good for shrinking images. Downsampling lowers resolution, speeding up subsequent
        processing steps but losing some detail.
        """
        if self.downsample_factor <= 1:
            return img

        h, w = img.shape[:2]
        new_size = (w // self.downsample_factor, h // self.downsample_factor)
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def upsample_image(
        self, img: np.ndarray, target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        After posterization on a downscaled image, this method scales the image back to the original dimensions
        using smooth cubic interpolation. This restores the image size for output, though some fine detail may
        be lost or smoothed out due to downsampling.
        """
        if self.downsample_factor <= 1:
            return img

        h, w = target_shape[:2]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

    @log_step("Preprocessing image (convert and blur)...")
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        First converts the image from uint8 (0-255) to floats (0-1) for numeric stability in processing.
        If blur_radius is set, applies Gaussian blur to smooth the image, which helps reduce noise and
        detail that might interfere with segmentation and palette extraction.
        """
        assert img.dtype == np.uint8, f"Expected uint8 image, got {img.dtype}"
        img_float = img.astype(np.float32) / 255

        if self.blur_radius > 0:
            ksize = int(
                BLUR_KERNEL_MULTIPLIER * round(self.blur_radius) + BLUR_KERNEL_OFFSET
            )
            # Ensure kernel size is odd and >= 3
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(3, ksize)

            # GaussianBlur accepts float32 input, no need to convert back and forth
            img_float = cv2.GaussianBlur(
                img_float, (ksize, ksize), sigmaX=self.blur_radius
            )

        return img_float

    @log_step(
        lambda self: f"Computing {self.num_superpixels} superpixels and {self.num_colors}-color palette..."
    )
    def _copmute_segments_and_palette(
        self, img_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Uses the SLIC algorithm to divide the image into superpixels, groups of pixels with similar color and proximity.
        Then computes average colors per segment and extracts a limited palette representing the image's main colors.
        """
        segments = self._compute_superpixels(img_rgb)
        palette_rgb, palette_lab = self._extract_palette(img_rgb, segments)
        for i, color in enumerate(palette_rgb):
            rgb_255 = (color * 255).astype(np.uint8)
            logger.info(f"  Color {i+1}: RGB{tuple(rgb_255.tolist())}")

        return segments, palette_rgb, palette_lab

    @log_step("Performing simple posterization...")
    def _simple_posterize(
        self, img_rgb: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Converts the image to Lab color space, then assigns each pixel to the closest palette color by Euclidean distance in Lab space.
        This step generates a simple, flat color version of the image without any edge or detail preservation.
        """
        img_lab = skcolor.rgb2lab(img_rgb)
        assignments = self._assign_pixels_to_palette(img_lab, palette_lab)
        return palette_rgb[assignments], img_lab

    @log_step("Preserving details with superpixel reconstruction and edge weighting...")
    def _preserve_detail(
        self,
        img_rgb: np.ndarray,
        img_lab: np.ndarray,
        segments: np.ndarray,
        palette_lab: np.ndarray,
        palette_rgb: np.ndarray,
        simple_post: np.ndarray,
    ) -> np.ndarray:
        """
        Computes average Lab colors per superpixel segment and finds weights describing how to mix palette colors to best approximate each segment.
        Expands these weights to pixels and reconstructs a smooth color version. Detects edges and uses them
        to blend between the simple posterization and weighted reconstruction, preserving sharp edges and fine details.
        """
        seg_means_lab, _ = self._compute_segment_means(img_lab, segments)
        seg_weights = self._compute_segment_weights(seg_means_lab, palette_lab)
        pixel_weights = self._expand_weights_to_pixels(seg_weights, segments)
        weighted_post = self._reconstruct_from_weights(pixel_weights, palette_rgb)

        edges = self._detect_edges((img_rgb * 255).astype(np.uint8))
        edge_weight = self.detail_blend_strength * edges[..., None]

        blended = simple_post * (1 - edge_weight) + weighted_post * edge_weight
        return np.clip(blended, 0, 1)

    @log_step("Applying smoothing filters...")
    def _smooth(self, img: np.ndarray) -> np.ndarray:
        """
        Runs multiple passes of bilateral filtering, which smooths colors but preserves edges by
        considering color similarity and spatial closeness. Then applies median blur to reduce noise further.
        This improves the visual quality of the posterized image, making colors blend nicely without losing important edges.
        """
        img_u8 = (img * 255).astype(np.uint8)

        passes = max(1, self.smoothing // 2)
        diameter = BILATERAL_DIAMETER_BASE + BILATERAL_DIAMETER_STEP * (
            self.smoothing // 2
        )
        sigma_color = SIGMA_COLOR_BASE + SIGMA_COLOR_STEP * self.smoothing
        sigma_space = SIGMA_SPACE_BASE + SIGMA_SPACE_STEP * self.smoothing

        for _ in range(passes):
            img_u8 = cv2.bilateralFilter(img_u8, diameter, sigma_color, sigma_space)

        median_ksize = MEDIAN_BLUR_BASE + MEDIAN_BLUR_STEP * ((self.smoothing - 1) // 2)
        img_u8 = cv2.medianBlur(img_u8, median_ksize)

        return img_u8.astype(np.float32) / 255

    @log_step("Final quantization to palette...")
    def _final_quantize(
        self, img: np.ndarray, palette_lab: np.ndarray, palette_rgb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recalculates pixel-to-palette assignments on the smoothed image, ensuring each pixel strictly uses one palette color.
        Also creates a one-hot weight matrix for each pixel's assigned palette color, which can be used to generate additive layers or masks.
        """
        lab = skcolor.rgb2lab(img)
        assignments = self._assign_pixels_to_palette(lab, palette_lab)
        posterized = palette_rgb[assignments]

        weights = np.eye(len(palette_rgb), dtype=np.float32)[assignments]
        return posterized, weights

    @log_step("Overlaying superpixel boundaries...")
    def _overlay_superpixel_boundaries(
        self,
        img: np.ndarray,
        segments: np.ndarray,
        boundary_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Overlays superpixel boundaries on the posterized image by detecting edges in the segment labels
        and drawing them as thin lines.
        """
        # Create overlay image - make a copy to avoid modifying the original
        overlay = img.copy()

        # Method 1: Use gradient-based boundary detection
        # Compute gradients to find boundaries between different segments
        grad_x = np.abs(np.diff(segments, axis=1))
        grad_y = np.abs(np.diff(segments, axis=0))

        # Create boundary mask
        boundaries = np.zeros_like(segments, dtype=bool)

        # Mark boundaries where gradients are non-zero
        boundaries[:, 1:] |= grad_x > 0
        boundaries[:, :-1] |= grad_x > 0
        boundaries[1:, :] |= grad_y > 0
        boundaries[:-1, :] |= grad_y > 0

        logger.info(
            f"Detected {np.sum(boundaries)} boundary pixels from {len(np.unique(segments))} segments"
        )

        # Apply boundary color where boundaries exist
        if np.sum(boundaries) > 0:
            for c in range(3):  # Apply to each color channel
                overlay[boundaries, c] = boundary_color[c]
            logger.info(f"Applied overlay to {np.sum(boundaries)} boundary pixels")
        else:
            logger.warning("No boundaries detected - overlay not applied")

        return overlay

    # Superpixels and segmentation helpers
    def _compute_superpixels(self, img: np.ndarray) -> np.ndarray:
        """
        Uses the skimage implementation of SLIC (Simple Linear Iterative Clustering) to divide the image into
        spatially compact and color-consistent segments.
        These segments help reduce complexity and preserve image structure in later steps.
        """
        return slic(
            img,
            n_segments=self.num_superpixels,
            compactness=self.superpixel_compactness,
            start_label=0,
        )

    def _compute_segment_means(
        self, lab_img: np.ndarray, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flattens the image and segment labels, then computes the mean Lab color of all pixels belonging to each segment.
        Also counts the number of pixels in each segment for filtering and weighting.
        """
        n_segments = segments.max() + 1
        flat_lab = lab_img.reshape(-1, 3)
        flat_segments = segments.flatten()

        counts = np.bincount(flat_segments, minlength=n_segments).astype(np.int64)
        sums = np.vstack(
            [
                np.bincount(flat_segments, weights=flat_lab[:, i], minlength=n_segments)
                for i in range(3)
            ]
        ).T
        means = np.zeros_like(sums)
        valid = counts > 0
        means[valid] = sums[valid] / counts[valid, None]

        return means, counts

    # Palette extraction and color assignment
    def _extract_palette(
        self,
        img: np.ndarray,
        segments: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters out very small segments, keeps only the biggest ones to create a palette of num_colors.
        Converts their average Lab colors back to RGB to use for color replacement during posterization.
        """
        lab_img = skcolor.rgb2lab(img)
        seg_means_lab, counts = self._compute_segment_means(lab_img, segments)

        total_pixels = img.shape[0] * img.shape[1]

        # Sort segments by descending pixel count
        sorted_idxs = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_idxs]
        cumulative_coverage = np.cumsum(sorted_counts) / total_pixels

        # Find minimal number of segments covering threshold or at least num_colors
        num_to_keep = np.searchsorted(cumulative_coverage, COVERAGE_THRESHOLD) + 1
        num_to_keep = max(num_to_keep, self.num_colors)

        if num_to_keep > len(counts):
            num_to_keep = len(counts)

        selected_idxs = sorted_idxs[:num_to_keep]

        valid = np.zeros_like(counts, dtype=bool)
        valid[selected_idxs] = True

        logger.info(
            f"Selected {num_to_keep} segments covering {cumulative_coverage[num_to_keep - 1]:.2%} of pixels "
            f"for palette extraction (requested {self.num_colors} colors)."
        )

        filtered_means = seg_means_lab[valid]
        filtered_counts = counts[valid]

        # If still have more segments than desired colors, pick top num_colors by pixel count
        if len(filtered_means) > self.num_colors:
            top_idxs = np.argsort(filtered_counts)[-self.num_colors :]
            filtered_means = filtered_means[top_idxs]

        palette_rgb = skcolor.lab2rgb(filtered_means[None, :, :]).reshape(-1, 3)
        return np.clip(palette_rgb, 0, 1), filtered_means

    def _assign_pixels_to_palette(
        self,
        img_lab: np.ndarray,
        palette_lab: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the Euclidean distance in Lab space between every pixel and each palette color,
        then picks the closest palette color for every pixel. Produces a 2D assignment map.
        """
        h, w = img_lab.shape[:2]
        pixels = img_lab.reshape(-1, 3)

        distances = np.linalg.norm(pixels[:, None, :] - palette_lab[None, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments.reshape(h, w)

    # Weight computations and reconstruction
    def _compute_segment_weights(
        self,
        seg_means_lab: np.ndarray,
        palette_lab: np.ndarray,
    ) -> np.ndarray:
        """
        For each segment's mean Lab color, solves a non-negative least squares problem to find weights for mixing palette
        colors to best represent that color.
        Normalizes weights to sum to 1 or falls back to nearest palette color if solution fails.
        """
        n_segments, k = seg_means_lab.shape[0], palette_lab.shape[0]
        weights = np.zeros((n_segments, k), dtype=np.float64)

        for i in range(n_segments):
            target = seg_means_lab[i]
            try:
                w, _ = nnls(palette_lab.T, target)
                s = w.sum()
                if s > 1e-8:
                    w /= s
                else:
                    idx = np.argmin(np.linalg.norm(palette_lab - target, axis=1))
                    w = np.zeros(k)
                    w[idx] = 1
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.error(
                    f"NNLS failed for segment {i} with error: {e}. Using nearest palette color fallback."
                )
                idx = np.argmin(np.linalg.norm(palette_lab - target, axis=1))
                w = np.zeros(k)
                w[idx] = 1
            weights[i] = w
        return weights

    def _expand_weights_to_pixels(
        self,
        segment_weights: np.ndarray,
        segments: np.ndarray,
    ) -> np.ndarray:
        """
        Maps segment-level color mixing weights down to individual pixels by using the segment labels,
        so each pixel inherits its segment's color mix weights.
        """
        h, w = segments.shape
        k = segment_weights.shape[1]

        flat_segments = segments.flatten()
        per_pixel_weights = segment_weights[flat_segments]
        return per_pixel_weights.reshape(h, w, k)

    def _reconstruct_from_weights(
        self,
        weights_per_pixel: np.ndarray,
        palette_rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Uses matrix multiplication of per-pixel weights with palette RGB colors to produce a smooth reconstructed
        image with continuous colors between palette entries. Clipped to valid color range.
        """
        reconstructed = weights_per_pixel @ palette_rgb
        return np.clip(reconstructed, 0, 1)

    def _detect_edges(
        self,
        img_rgb_u8: np.ndarray,
    ) -> np.ndarray:
        """
        Detect edges using Canny on downscaled grayscale blurred image.
        Thresholds scale with edge_threshold.
        """
        small = cv2.resize(
            img_rgb_u8,
            (0, 0),
            fx=EDGE_DETECTION_SCALE,
            fy=EDGE_DETECTION_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        low = max(1, int(CANNY_LOW_THRESHOLD_BASE * self.edge_threshold))
        high = max(
            low + CANNY_THRESHOLD_MARGIN,
            int(CANNY_HIGH_THRESHOLD_BASE * self.edge_threshold),
        )

        edges_small = cv2.Canny(blurred, low, high) / 255.0
        edges = cv2.resize(
            edges_small,
            (img_rgb_u8.shape[1], img_rgb_u8.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return edges

    # Output and visualization

    def _create_palette_swatch(
        self,
        palette_rgb: np.ndarray,
        height: int = 64,
    ) -> np.ndarray:
        """
        Creates a horizontal image showing all palette colors as blocks.
        """
        n = palette_rgb.shape[0]
        width = n * height
        swatch = np.zeros((height, width, 3), dtype=np.float32)
        for i, color in enumerate(palette_rgb):
            swatch[:, i * height : (i + 1) * height, :] = color[None, None, :]
        return swatch

    def _save_additive_layers(
        self,
        mixing_weights: np.ndarray,
        palette_rgb: np.ndarray,
        prefix: Path,
    ) -> None:
        """
        Saves one image per palette color showing that color's contribution and a mask.
        """
        if not isinstance(prefix, Path):
            prefix = Path(prefix)

        n_colors = palette_rgb.shape[0]
        for i in range(n_colors):
            alpha = mixing_weights[:, :, i]
            layer = np.zeros((*alpha.shape, 4), dtype=np.float32)
            layer[..., :3] = palette_rgb[i]
            layer[..., 3] = alpha
            rgba_img = (np.clip(layer, 0, 1) * 255).astype(np.uint8)

            file_path = prefix.with_name(f"{prefix.stem}-{i}.png")
            Image.fromarray(rgba_img, mode="RGBA").save(file_path)

            mask = (alpha > 1e-6).astype(np.uint8) * 255
            mask_rgb = np.stack([mask] * 3, axis=2)
            mask_path = prefix.with_name(f"{prefix.stem}-{i}-mask.png")
            Image.fromarray(mask_rgb, mode="RGB").save(mask_path)

        logger.info(f"Additive mixing layers saved with prefix '{prefix}'")


def posterize(
    input_path: Path | Image.Image | str,
    output_path: Path | str,
    *,
    num_colors: int = 52,
    blur_radius: float = 1.5,
    edge_threshold: float = 0.1,
    downsample_factor: int = 1,
    preserve_edges: bool = True,
    num_superpixels: int = 4000,
    superpixel_compactness: float = 15.0,
    detail_blend_strength: float = 0.05,
    smoothing: int = 5,
    palette_path: Path | str | None = None,
    mixing_prefix: Path | str | None = None,
    quality: int = 95,
    overlay_superpixels: bool = False,
):
    """
    Run the posterization pipeline on an image file or loaded image.
    """
    posterizer = SLICPosterizer(
        num_colors=num_colors,
        blur_radius=blur_radius,
        edge_threshold=edge_threshold,
        downsample_factor=downsample_factor,
        preserve_edges=preserve_edges,
        num_superpixels=num_superpixels,
        superpixel_compactness=superpixel_compactness,
        detail_blend_strength=detail_blend_strength,
        smoothing=smoothing,
        overlay_superpixels=overlay_superpixels,
    )

    posterizer.posterize(
        input_path,
        output_path,
        palette_path=palette_path,
        mixing_prefix=mixing_prefix,
        quality=quality,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SLIC Posterizer - SLIC-Based Artistic Posterization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", nargs="?", help="Input image path", default=None)
    parser.add_argument("output", nargs="?", help="Output posterized image path")

    parser.add_argument(
        "-c", "--colors", type=int, default=64, help="Number of colors in palette"
    )
    parser.add_argument(
        "-b", "--blur", type=float, default=1, help="Blur radius for smoothing"
    )
    parser.add_argument(
        "-s",
        "--smoothing",
        type=int,
        default=3,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Smoothing strength level (1-10)",
    )
    parser.add_argument(
        "-S", "--superpixels", type=int, default=4500, help="Number of superpixels"
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=15.0,
        help="SLIC superpixel compactness parameter",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay superpixel boundaries on the final image",
    )
    parser.add_argument(
        "-m", "--mixing", help="Output prefix for additive mixing layers"
    )
    parser.add_argument("-p", "--palette", help="Output path for palette swatch")
    parser.add_argument(
        "-e",
        "--edge-threshold",
        type=float,
        default=0.1,
        help="Edge detection threshold",
    )
    parser.add_argument(
        "-d", "--downsample", type=int, default=1, help="Downsample factor (>=1)"
    )
    parser.add_argument(
        "--detail-blend",
        type=float,
        default=0.1,
        help="Blend factor for detail preservation",
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="JPEG quality (if saving JPEG)"
    )
    parser.add_argument(
        "--no-edge-preserve", action="store_true", help="Disable edge preservation"
    )

    args = parser.parse_args()

    if not sys.stdin.isatty():
        # Reading from stdin
        if not args.output:
            if args.input:
                output_path = args.input
            else:
                parser.error(
                    "Output path required when reading from stdin without input path."
                )
        else:
            output_path = args.output
        input_data = Image.open(BytesIO(sys.stdin.buffer.read()))
    else:
        if not args.input or not args.output:
            parser.error(
                "Both input and output paths are required unless piping data into stdin."
            )
        input_data = args.input
        output_path = args.output

    posterize(
        input_path=input_data,
        output_path=output_path,
        num_colors=args.colors,
        blur_radius=args.blur,
        edge_threshold=args.edge_threshold,
        downsample_factor=args.downsample,
        preserve_edges=not args.no_edge_preserve,
        num_superpixels=args.superpixels,
        superpixel_compactness=args.compactness,
        detail_blend_strength=args.detail_blend,
        smoothing=args.smoothing,
        palette_path=args.palette,
        mixing_prefix=args.mixing,
        quality=args.quality,
        overlay_superpixels=args.overlay,
    )


if __name__ == "__main__":
    main()

