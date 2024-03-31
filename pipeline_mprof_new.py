"""This file contains the pipeline that will be profiled by the memory_profiler package. This file implements the new version of CAIMAN (as opposed to pipeline_mprof_old.py implementing the old version of CAIMAN). Because memory_profiler requires decorators to help it run, this code cannot be imported to other files without error (unless you were to implement something that could ignore the decorators). Use pipeline_memray_new.py for that function (or better practice probably would be to make a new file that is more readable).
"""

import pickle
import pandas as pd
import numpy as np
from scipy.signal.windows import tukey
from scipy import interpolate as interp
from scipy import ndimage
import scipy
from scipy.ndimage import percentile_filter
from caiman.source_extraction.cnmf import deconvolution
from scipy import fftpack
import pyfftw
from imreg_dft import utils
from caiman import load_memmap, components_evaluation
import time, sys, os, glob
import multiprocessing as mp
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf import (
    map_reduce,
    merging,
    initialization,
    pre_processing,
    spatial,
    temporal,
)


@profile
def pipeline_with_dataload(
    file_location,
    scan_idx=0,
    temporal_fill_fraction=1,
    in_place=False,
    fps=15,
    should_perform_raster_correction=True,
    should_perform_motion_correction=True,
    should_extract_masks=True,
    should_deconvolve=False,
):
    #data = load_pickle(file_location)
    #scan = extract_scan_from_pandas(data, scan_idx)
    mmap_filename = 'segmentation/data/caiman/caiman_d1_240_d2_240_d3_1_order_C_frames_61717_.mmap'
    mmap_scan, (image_height, image_width), num_frames = load_memmap(mmap_filename)
    scan = np.array(mmap_scan)
    scan = scan.reshape(image_height, image_width, num_frames, order='F')
    pipeline_output = pipeline(
        scan,
        temporal_fill_fraction,
        in_place,
        fps,
        should_perform_raster_correction,
        should_perform_motion_correction,
        should_extract_masks,
        should_deconvolve,
        mmap_scan
    )
    return pipeline_output


@profile
def load_pickle(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data


@profile
def extract_scan_from_pandas(data, scan_idx):
    return data.loc[scan_idx, "mini_scan"]


@profile
def pipeline(
    scan,
    temporal_fill_fraction=1,
    in_place=False,
    fps=15,
    should_perform_raster_correction=True,
    should_perform_motion_correction=True,
    should_extract_masks=True,
    should_deconvolve=True,
    mmap_scan=None
):

    if should_perform_raster_correction:
        print("Performing raster correction...")
        scan = perform_raster_correction(scan, temporal_fill_fraction, in_place)
    if should_perform_motion_correction:
        print('Performing motion correction...')
        scan = perform_motion_correction(scan, in_place)

    if should_extract_masks:
        print('Performing segmentation...')
        # Save as memory mapped file in F order (that's how caiman wants it)
        #mmap_filename = save_as_memmap(scan, base_name="data/caiman/caiman").filename

        # 'Load' scan
        #mmap_scan, (image_height, image_width), num_frames = load_memmap(mmap_filename)
        profiling_params = params = {'num_background_components': 1,
             'merge_threshold': 0.7,
             'fps': 8.3091,
             'init_on_patches': True,
             'proportion_patch_overlap': 0.2,
             'num_components_per_patch': 6,
             'init_method': 'greedy_roi',
             'patch_size': [20.0, 20.0],
             'soma_diameter': [3.2, 3.2],
             'num_processes': 8,
             'num_pixels_per_process': 10000}
    
        (
            masks,
            traces,
            background_masks,
            background_traces,
            raw_traces,
        ) = extract_masks_adapted(scan, mmap_scan, **params)

        if should_deconvolve:
            print('Performing deconvolution...')
            spike_traces = []
            AR_coeff_all = []
            for trace in traces:
                spike_trace, AR_coeffs = deconvolve_detrended(trace, fps)
                spike_traces.append(spike_trace)
                AR_coeff_all.append(AR_coeffs)
            return (
                spike_traces,
                AR_coeff_all,
                masks,
                traces,
                background_masks,
                background_traces,
                raw_traces,
            )
        return masks, traces, background_masks, background_traces, raw_traces


@profile
def anscombe_transform_raster(mini_scan):
    return 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # anscombe transform


@profile
def anscombe_transform_motion(mini_scan):
    return 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # anscombe transform


def select_middle_frames(scan, skip_rows=0, skip_cols=0):
    # Load some frames from the middle of the scan
    num_frames = scan.shape[-1]
    middle_frame = int(np.floor(num_frames / 2))
    frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
    last_row = -scan.shape[0] if skip_rows == 0 else skip_rows
    last_col = -scan.shape[1] if skip_cols == 0 else skip_cols
    mini_scan = scan[skip_rows:-last_row, skip_cols:-last_col, frames]

    return mini_scan


@profile
def raster_template_prep(scan):
    # Load middle 2000 frames from the scan
    mini_scan = select_middle_frames(scan)

    # get the height and width of the scan
    height, width = mini_scan.shape[:2]

    return mini_scan, height, width


@profile
def compute_raster_template(scan):
    mini_scan, height, width = raster_template_prep(scan)

    # Create template (average frame tapered to avoid edge artifacts)
    taper = np.sqrt(
        np.outer(
            tukey(height, 0.4),
            tukey(width, 0.4),
        )
    )
    anscombed = anscombe_transform_raster(mini_scan)
    template = np.mean(anscombed, axis=-1) * taper

    return template


@profile
def raster_phase_prep(image, temporal_fill_fraction):
    # Make sure image has even number of rows (so number of even and odd rows is the same)
    image = image[:-1] if image.shape[0] % 2 == 1 else image

    # Get some params
    image_height, image_width = image.shape
    skip_rows = round(image_height * 0.05)  # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10)  # so do columns

    # Create images with even and odd rows
    even_rows = image[::2][skip_rows:-skip_rows]
    odd_rows = image[1::2][skip_rows:-skip_rows]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]
    # sin_index = np.sin(scan_angles)

    even_interp, odd_interp = create_interp_functions(scan_angles, even_rows, odd_rows)

    return image, skip_rows, skip_cols, scan_angles, even_interp, odd_interp


@profile
def create_interp_functions(scan_angles, even_rows, odd_rows):
    even_interp = interp.interp1d(scan_angles, even_rows, fill_value="extrapolate")
    odd_interp = interp.interp1d(scan_angles, odd_rows, fill_value="extrapolate")
    return even_interp, odd_interp


@profile
def compute_raster_phase(image: np.array, temporal_fill_fraction: float) -> float:
    """Compute raster correction for bidirectional resonant scanners.

    It shifts the even and odd rows of the image in the x axis to find the scan angle
    that aligns them better. Positive raster phase will shift even rows to the right and
    odd rows to the left (assuming first row is row 0).

    :param np.array image: The image to be corrected.
    :param float temporal_fill_fraction: Fraction of time during which the scan is
        recording a line against the total time per line.

    :return: An angle (in radians). Estimate of the mismatch angle between the expected
         initial angle and the one recorded.
    :rtype: float
    """
    # Make sure image has even number of rows (so number of even and odd rows is the same)
    (
        image,
        skip_rows,
        skip_cols,
        scan_angles,
        even_interp,
        odd_interp,
    ) = raster_phase_prep(image, temporal_fill_fraction)

    # Greedy search for the best raster phase: starts at coarse estimates and refines them
    angle_shift = 0
    for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        angle_shifts = angle_shift + scale * np.linspace(-9, 9, 19)
        match_values = []
        for new_angle_shift in angle_shifts:
            shifted_evens = even_interp(scan_angles + new_angle_shift)
            shifted_odds = odd_interp(scan_angles - new_angle_shift)
            match_values.append(
                np.sum(
                    shifted_evens[:, skip_cols:-skip_cols]
                    * shifted_odds[:, skip_cols:-skip_cols]
                )
            )
        angle_shift = angle_shifts[np.argmax(match_values)]

    return angle_shift


@profile
def make_copy_correct_raster(scan):
    return scan.copy()


@profile
def correct_raster_prep(scan, temporal_fill_fraction, in_place):
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException("Scan needs to be a numpy array.")
    if scan.ndim < 2:
        raise PipelineException("Scan with less than 2 dimensions.")

    # Assert scan is float
    if not np.issubdtype(scan.dtype, np.floating):
        print("Warning: Changing scan type from", str(scan.dtype), "to np.float32")
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = make_copy_correct_raster(
            scan
        )  # copy it anyway preserving the original float dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]

    return reshaped_scan, num_images, original_shape, scan_angles


def correction_post(reshaped_scan, original_shape):
    scan = np.reshape(reshaped_scan, original_shape)
    return scan


@profile
def correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=True):
    """Raster correction for resonant scans.

    Corrects multi-photon images in n-dimensional scans. Positive raster phase shifts
    even lines to the left and odd lines to the right. Negative raster phase shifts even
    lines to the right and odd lines to the left.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
        Works for 2-dimensions and up, usually (image_height, image_width, num_frames).
    :param float raster_phase: Angle difference between expected and recorded scan angle.
    :param float temporal_fill_fraction: Ratio between active acquisition and total
        length of the scan line.
    :param bool in_place: If True (default), the original array is modified in place.

    :return: Raster-corrected scan.
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.float32.

    :raises: PipelineException
    """
    reshaped_scan, num_images, original_shape, scan_angles = correct_raster_prep(
        scan, temporal_fill_fraction, in_place
    )

    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]

        (
            even_interp_function,
            odd_interp_function,
        ) = create_interp_functions_raster_correction(scan_angles, image, in_place)

        # Correct even rows of the image (0, 2, ...)
        reshaped_scan[::2, :, i] = even_interp_function(scan_angles + raster_phase)

        # Correct odd rows of the image (1, 3, ...)
        reshaped_scan[1::2, :, i] = odd_interp_function(scan_angles - raster_phase)

    scan = correction_post(reshaped_scan, original_shape)
    return scan


def create_interp_functions_raster_correction(scan_angles, image, in_place):
    # Correct even rows of the image (0, 2, ...)
    even_interp_function = interp.interp1d(
        scan_angles,
        image[::2, :],
        bounds_error=False,
        fill_value=0,
        copy=(not in_place),
    )
    odd_interp_function = interp.interp1d(
        scan_angles,
        image[1::2, :],
        bounds_error=False,
        fill_value=0,
        copy=(not in_place),
    )

    return even_interp_function, odd_interp_function


@profile
def perform_raster_correction(scan, temporal_fill_fraction=1, in_place=False):
    raster_template = compute_raster_template(scan)
    raster_phase = compute_raster_phase(raster_template, temporal_fill_fraction)
    raster_corrected_scan = correct_raster(
        scan,
        raster_phase,
        temporal_fill_fraction=temporal_fill_fraction,
        in_place=in_place,
    )

    return raster_corrected_scan


@profile
def motion_template_prep(scan):
    ## Get needed info
    px_height, px_width = scan.shape[:2]
    skip_rows = int(
        round(px_height * 0.10)
    )  # we discard some rows/cols to avoid edge artifacts
    skip_cols = int(round(px_width * 0.10))

    ## Select template source
    # Default behavior: use middle 2000 frames as template source
    mini_scan = select_middle_frames(scan, skip_rows, skip_cols)

    return mini_scan


@profile
def create_motion_template(scan):
    """
    Creates the template all frames are compared against to determine the
    amount of motion that occured. Exclusively used for the first iteration
    of motion correction.
    """

    mini_scan = motion_template_prep(scan)

    # Create template
    mini_scan = anscombe_transform_motion(mini_scan)
    template = np.mean(mini_scan, axis=-1).squeeze()

    # Apply spatial filtering (if needed)
    template = ndimage.gaussian_filter(template, 0.7)  # **
    # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
    # ** Small amount of gaussian smoothing to get rid of high frequency noise

    return template


@profile
def create_copy_motion_shifts_prep(scan, in_place):
    if not in_place:
        scan = scan.copy()
    return scan


@profile
def motion_shifts_prep(scan, template, in_place, num_threads):
    # Add third dimension if scan is a single image
    # if scan.ndim == 2:
    #     scan = np.expand_dims(scan, -1)
    scan = create_copy_motion_shifts_prep(scan, in_place)

    # Get some params
    image_height, image_width, num_frames = scan.shape
    skip_rows = int(
        round(image_height * 0.10)
    )  # we discard some rows/cols to avoid edge artifacts
    skip_cols = int(round(image_width * 0.10))
    scan = scan[
        skip_rows:-skip_rows,
        skip_cols:-skip_cols,
        :,
    ]
    (
        image_height,
        image_width,
        num_frames,
    ) = scan.shape  # recalculate after removing edge rows/cols
    taper = np.outer(tukey(image_height, 0.2), tukey(image_width, 0.2))

    template_freq, abs_template_freq, eps, fft, ifft = fftw_prep(
        image_height, image_width, template, taper, num_threads, in_place
    )

    return (
        scan,
        num_frames,
        taper,
        template_freq,
        abs_template_freq,
        eps,
        fft,
        ifft,
        image_height,
        image_width,
    )


def fftw_prep(image_height, image_width, template, taper, num_threads, in_place):
    # Prepare fftw
    frame = pyfftw.empty_aligned((image_height, image_width), dtype="complex64")
    fft = pyfftw.builders.fft2(
        frame, threads=num_threads, overwrite_input=in_place, avoid_copy=True
    )
    ifft = pyfftw.builders.ifft2(
        frame, threads=num_threads, overwrite_input=in_place, avoid_copy=True
    )

    # Get fourier transform of template
    template_freq = fft(template * taper).conj()  # we only need the conjugate
    abs_template_freq = abs(template_freq)
    eps = abs_template_freq.max() * 1e-15

    return template_freq, abs_template_freq, eps, fft, ifft


def compute_cross_power(
    fft, scan, i, taper, template_freq, abs_template_freq, eps, ifft
):
    image_freq = fft(scan[:, :, i] * taper)
    cross_power = (image_freq * template_freq) / (
        abs(image_freq) * abs_template_freq + eps
    )
    shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))
    return shifted_cross_power


def map_deviations(y_shifts, x_shifts, i, shifts, image_height, image_width):
    y_shifts[i] = shifts[0] - image_height // 2
    x_shifts[i] = shifts[1] - image_width // 2
    return y_shifts, x_shifts


def get_best_shift(shifted_cross_power):
    shifts = np.unravel_index(np.argmax(shifted_cross_power), shifted_cross_power.shape)
    shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)
    return shifts


@profile
def compute_motion_shifts(scan, template, in_place=True, num_threads=8):
    """Compute shifts in y and x for rigid subpixel motion correction.

    Returns the number of pixels that each image in the scan was to the right (x_shift)
    or below (y_shift) the template. Negative shifts mean the image was to the left or
    above the template.

    :param np.array scan: 2 or 3-dimensional scan (image_height, image_width[, num_frames]).
    :param np.array template: 2-d template image. Each frame in scan is aligned to this.
    :param bool in_place: Whether the scan can be overwritten.
    :param int num_threads: Number of threads used for the ffts.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the y, x motion shifts.

    ..note:: Based in imreg_dft.translation().
    """
    (
        scan,
        num_frames,
        taper,
        template_freq,
        abs_template_freq,
        eps,
        fft,
        ifft,
        image_height,
        image_width,
    ) = motion_shifts_prep(scan, template, in_place, num_threads)

    # Compute subpixel shifts per image
    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):
        # Compute correlation via cross power spectrum
        shifted_cross_power = compute_cross_power(
            fft, scan, i, taper, template_freq, abs_template_freq, eps, ifft
        )

        # Get best shift
        shifts = get_best_shift(shifted_cross_power)

        # Map back to deviations from center
        y_shifts, x_shifts = map_deviations(
            y_shifts, x_shifts, i, shifts, image_height, image_width
        )

    return y_shifts, x_shifts


@profile
def make_copy_correct_motion_prep(scan):
    return scan.copy()


@profile
def make_two_copy_correct_motion(scan1, scan2):
    return scan1.copy(), scan2.copy()


@profile
def correct_motion_prep(scan, y_shifts, x_shifts, in_place):
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException("Scan needs to be a numpy array.")
    if scan.ndim < 2:
        raise PipelineException("Scan with less than 2 dimensions.")
    if np.ndim(y_shifts) != 1 or np.ndim(x_shifts) != 1:
        raise PipelineException(
            "Dimension of one or both motion arrays differs from 1."
        )
    if len(x_shifts) != len(y_shifts):
        raise PipelineException("Length of motion arrays differ.")

    # Assert scan is float (integer precision is not good enough)
    if not np.issubdtype(scan.dtype, np.floating):
        print("Warning: Changing scan type from", str(scan.dtype), "to np.float32")
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = make_copy_correct_motion_prep(
            scan
        )  # copy it anyway preserving the original dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Reshape input (to deal with more than 2-D volumes)
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    if reshaped_scan.shape[-1] != len(x_shifts):
        raise PipelineException("Scan and motion arrays have different dimensions")

    # Ignore NaN values (present in some older data)
    y_clean, x_clean = make_two_copy_correct_motion(y_shifts, x_shifts)
    y_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0
    x_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0

    return y_clean, x_clean, reshaped_scan, original_shape


def make_copy_correct_motion(image):
    return image.copy()


def motion_correction_shift(image, y_shift, x_shift, reshaped_scan, i):
    ndimage.shift(image, (-y_shift, -x_shift), order=1, output=reshaped_scan[:, :, i])


@profile
def correct_motion(scan, x_shifts, y_shifts, in_place=True):
    """Motion correction for multi-photon scans.

    Shifts each image in the scan x_shift pixels to the left and y_shift pixels up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
        Works for 2-dimensions and up, usually (image_height, image_width, num_frames).
    :param list/np.array x_shifts: 1-d array with x motion shifts for each image.
    :param list/np.array y_shifts: 1-d array with x motion shifts for each image.
    :param bool in_place: If True (default), the original array is modified in place.

    :return: Motion corrected scan
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.float32.

    :raises: PipelineException
    """
    y_clean, x_clean, reshaped_scan, original_shape = correct_motion_prep(
        scan, y_shifts, x_shifts, in_place
    )

    # Shift each frame
    for i, (y_shift, x_shift) in enumerate(zip(y_clean, x_clean)):
        image = make_copy_correct_motion(reshaped_scan[:, :, i])
        motion_correction_shift(image, y_shift, x_shift, reshaped_scan, i)

    scan = correction_post(reshaped_scan, original_shape)
    return scan


@profile
def perform_motion_correction(scan, in_place=False):
    motion_template = create_motion_template(scan)
    y_shifts, x_shifts = compute_motion_shifts(scan, motion_template, in_place)
    motion_corrected_scan = correct_motion(scan, x_shifts, y_shifts, in_place)

    return motion_corrected_scan


@profile
def log(*messages):
    """Simple logging function."""
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True, file=sys.__stdout__)


def _greedyROI(
    scan, num_components=200, neuron_size=(11, 11), num_background_components=1
):
    """Initialize components by searching for gaussian shaped, highly active squares.
    #one by one by moving a gaussian window over every pixel and
    taking the highest activation as the center of the next neuron.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: The desired number of components.
    :param (float, float) neuron_size: Expected size of the somas in pixels (y, x).
    :param int num_background_components: Number of components that model the background.
    """
    from scipy import ndimage

    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Get the gaussian kernel
    gaussian_stddev = (
        np.array(neuron_size) / 4
    )  # entire neuron in four standard deviations
    gaussian_kernel = _gaussian2d(gaussian_stddev)

    # Create residual scan (scan minus background)
    residual_scan = scan - np.mean(scan, axis=(0, 1))  # image-wise brightness
    background = ndimage.gaussian_filter(np.mean(residual_scan, axis=-1), neuron_size)
    residual_scan -= np.expand_dims(background, -1)

    # Create components
    masks = np.zeros([image_height, image_width, num_components], dtype=np.float32)
    traces = np.zeros([num_components, num_frames], dtype=np.float32)
    mean_frame = np.mean(residual_scan, axis=-1)
    for i in range(num_components):
        # Get center of next component
        neuron_locations = ndimage.gaussian_filter(mean_frame, gaussian_stddev)
        y, x = np.unravel_index(
            np.argmax(neuron_locations), [image_height, image_width]
        )

        # Compute initial trace (bit messy because of edges)
        half_kernel = np.fix(np.array(gaussian_kernel.shape) / 2).astype(np.int32)
        big_yslice = slice(max(y - half_kernel[0], 0), y + half_kernel[0] + 1)
        big_xslice = slice(max(x - half_kernel[1], 0), x + half_kernel[1] + 1)
        kernel_yslice = slice(
            max(0, half_kernel[0] - y),
            None
            if image_height > y + half_kernel[0]
            else image_height - y - half_kernel[0] - 1,
        )
        kernel_xslice = slice(
            max(0, half_kernel[1] - x),
            None
            if image_width > x + half_kernel[1]
            else image_width - x - half_kernel[1] - 1,
        )
        cropped_kernel = gaussian_kernel[kernel_yslice, kernel_xslice]
        trace = np.average(
            residual_scan[big_yslice, big_xslice].reshape(-1, num_frames),
            weights=cropped_kernel.ravel(),
            axis=0,
        )

        # Get mask and trace using 1-rank NMF
        half_neuron = np.fix(np.array(neuron_size) / 2).astype(np.int32)
        yslice = slice(max(y - half_neuron[0], 0), y + half_neuron[0] + 1)
        xslice = slice(max(x - half_neuron[1], 0), x + half_neuron[1] + 1)
        mask, trace = _rank1_NMF(residual_scan[yslice, xslice], trace)

        # Update residual scan
        neuron_activity = np.expand_dims(mask, -1) * trace
        residual_scan[yslice, xslice] -= neuron_activity
        mean_frame[yslice, xslice] = np.mean(residual_scan[yslice, xslice], axis=-1)

        # Store results
        masks[yslice, xslice, i] = mask
        traces[i] = trace

    # Create background components
    residual_scan += np.mean(scan, axis=(0, 1))  # add back overall brightness
    residual_scan += np.expand_dims(background, -1)  # and background
    if num_background_components == 1:
        background_masks = np.expand_dims(np.mean(residual_scan, axis=-1), axis=-1)
        background_traces = np.expand_dims(np.mean(residual_scan, axis=(0, 1)), axis=0)
    else:
        from sklearn.decomposition import NMF

        print(
            "Warning: Fitting more than one background component uses scikit-learn's "
            "NMF and may take some time."
            ""
        )
        model = NMF(num_background_components, random_state=123, verbose=True)

        flat_masks = model.fit_transform(residual_scan.reshape(-1, num_frames))
        background_masks = flat_masks.reshape([image_height, image_width, -1])
        background_traces = model.components_

    return masks, traces, background_masks, background_traces


def _gaussian2d(stddev, truncate=4):
    """Creates a 2-d gaussian kernel truncated at 4 standard deviations (8 in total).

    :param (float, float) stddev: Standard deviations in y and x.
    :param float truncate: Number of stddevs at each side of the kernel.

    ..note:: Kernel sizes will always be odd.
    """
    from matplotlib import mlab

    half_kernel = np.round(stddev * truncate)  # kernel_size = 2 * half_kernel + 1
    y, x = np.meshgrid(
        np.arange(-half_kernel[0], half_kernel[0] + 1),
        np.arange(-half_kernel[1], half_kernel[1] + 1),
    )
    kernel = mlab.bivariate_normal(x, y, sigmay=stddev[0], sigmax=stddev[1])
    return kernel


# Based on caiman.source_extraction.cnmf.initialization.finetune()
def _rank1_NMF(scan, trace, num_iterations=5):
    num_frames = scan.shape[-1]
    for i in range(num_iterations):
        mask = np.maximum(np.dot(scan, trace), 0)
        mask = mask * np.sum(mask) / np.sum(mask**2)
        trace = np.average(scan.reshape(-1, num_frames), weights=mask.ravel(), axis=0)
    return mask, trace


@profile
def save_as_memmap(scan, base_name="caiman", chunk_size=5000):
    """Save the scan as a memory mapped file as expected by caiman

    :param np.array scan: Scan to save shaped (image_height, image_width, num_frames)
    :param string base_name: Base file name for the scan. No underscores.
    :param int chunk_size: Write the mmap_scan chunk frames at a time. Memory efficient.

    :returns: Filename of the mmap file.
    :rtype: string
    """
    # Get some params
    image_height, image_width, num_frames = scan.shape
    num_pixels = image_height * image_width

    # Build filename
    filename = "{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap".format(
        base_name, image_height, image_width, num_frames
    )

    # Create memory mapped file
    mmap_scan = np.memmap(
        filename, mode="w+", shape=(num_pixels, num_frames), dtype=np.float32
    )
    for i in range(0, num_frames, chunk_size):
        chunk = scan[..., i : i + chunk_size].reshape((num_pixels, -1), order="F")
        mmap_scan[:, i : i + chunk_size] = chunk
    mmap_scan.flush()

    return mmap_scan


@profile
def extract_masks_adapted(
    scan,
    mmap_scan,
    num_components=200,
    num_background_components=1,
    merge_threshold=0.8,
    init_on_patches=True,
    init_method="greedy_roi",
    soma_diameter=(14, 14),
    snmf_alpha=0.5,
    patch_size=(50, 50),
    proportion_patch_overlap=0.2,
    num_components_per_patch=5,
    num_processes=8,
    num_pixels_per_process=5000,
    fps=15,
):
    """Extract masks from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find spatial components (masks)
    and their fluorescence traces in a scan. Default values work well for somatic scans.

    Performed operations are:
        [Initialization on full image | Initialization on patches -> merge components] ->
        spatial update -> temporal update -> merge components -> spatial update ->
        temporal update

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param np.memmap mmap_scan: 2-d scan (image_height * image_width, num_frames)
    :param int num_components: An estimate of the number of spatial components in the scan
    :param int num_background_components: Number of components to model the background.
    :param int merge_threshold: Maximal temporal correlation allowed between the activity
        of overlapping components before merging them.
    :param bool init_on_patches: If True, run the initialization methods on small patches
        of the scan rather than on the whole image.
    :param string init_method: Initialization method for the components.
        'greedy_roi': Look for a gaussian-shaped patch, apply rank-1 NMF, store
            components, calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
    :param (float, float) soma_diameter: Estimated neuron size in y and x (pixels). Used
        in'greedy_roi' initialization to search for neurons of this size.
    :param int snmf_alpha: Regularization parameter (alpha) for sparse NMF (if used).
    :param (float, float) patch_size: Size of the patches in y and x (pixels).
    :param float proportion_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%).
    :param int num_components_per_patch: Number of components per patch (used if
        init_on_patches=True)
    :param int num_processes: Number of processes to run in parallel. None for as many
        processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
        iteration.
    :param fps: Frame rate. Used for temporal downsampling and to remove bad components.

    :returns: Weighted masks (image_height x image_width x num_components). Inferred
        location of each component.
    :returns: Denoised fluorescence traces (num_components x num_frames).
    :returns: Masks for background components (image_height x image_width x
        num_background_components).
    :returns: Traces for background components (image_height x image_width x
        num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames). Fluorescence of each
        component in the scan minus activity from other components and background.

    ..warning:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Better results if scans are nonnegative.
    """
    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Start processes
    log("Starting {} processes...".format(num_processes))
    pool = mp.Pool(processes=num_processes)

    # Initialize components
    log("Initializing components...")
    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

        # Make sure they are integers
        patch_size = np.array(patch_size)
        half_patch_size = np.int32(np.round(patch_size / 2))
        num_components_per_patch = int(round(num_components_per_patch))
        patch_overlap = np.int32(np.round(patch_size * proportion_patch_overlap))

        # Create options dictionary (needed for run_CNMF_patches)
        options = {
            "patch": {
                "nb_batch": num_background_components,
                "only_init": True,
                "remove_very_bad_comps": False,
                "rf": half_patch_size,
                "stride": patch_overlap,
            },  # remove_very_bads_comps unnecesary (same as default)
            "preprocess": {
                "check_nan": False
            },  # check_nan is unnecessary (same as default value)
            "spatial": {
                "nb": num_background_components,
                "n_pixels_per_process": num_pixels_per_process,
            },  # nb is unnecessary, it is pased to the function and in init_params
            "temporal": {
                "p": 0,
                "block_size_temp": 10000,
                "method_deconvolution": "cvxpy",
            },
            "init": {
                "K": num_components_per_patch,
                "gSig": np.array(soma_diameter) / 2,
                "method_init": init_method,
                "alpha_snmf": snmf_alpha,
                "nb": num_background_components,
                "ssub": 1,
                "tsub": max(int(fps / 2), 1),
                "normalize_init": True,
                "rolling_sum": True,
                "rolling_length": 100,
            },
            # gSiz, ssub, tsub, options_local_NMF, normalize_init, rolling_sum unnecessary (same as default values)
            "merging": {"merge_thr": 0.8},
        }

        params = CNMFParams()
        for key in options:
            params.set(key, options[key])

        # Initialize per patch
        res = map_reduce.run_CNMF_patches(
            mmap_scan.filename,
            (image_height, image_width, num_frames),
            params,
            dview=pool,
            memory_fact=params.get("patch", "memory_fact"),
            gnb=params.get("init", "nb"),
            border_pix=params.get("patch", "border_pix"),
            low_rank_background=params.get("patch", "low_rank_background"),
            del_duplicates=params.get("patch", "del_duplicates"),
        )  # indices=[slice(None)]*3
        initial_A, initial_C, YrA, initial_b, initial_f, pixels_noise, _ = res

        # bl, c1, g, neurons_noise = None, None, None, None

        # Merge spatially overlapping components
        merged_masks = ["dummy"]
        while len(merged_masks) > 0:
            res = merging.merge_components(
                mmap_scan,
                initial_A,
                initial_b,
                initial_C,
                YrA,
                initial_f,
                initial_C,
                pixels_noise,
                params.get_group("temporal"),
                params.get_group("spatial"),
                dview=pool,
                thr=params.get("merging", "merge_thr"),
                mx=np.Inf,
            )  # ,
            # bl=bl,
            # c1=c1
            # sn=neurons_noise,
            # g=g
            # )
            (
                initial_A,
                initial_C,
                num_components,
                merged_masks,
                S,
                bl,
                c1,
                neurons_noise,
                g,
                empty_merged,
                YrA,
            ) = res

        # Delete log files (one per patch)
        log_files = glob.glob("caiman*_LOG_*")
        for log_file in log_files:
            os.remove(log_file)

    # TODO: GET THIS ELSE BLOCK WORKING
    else:
        from scipy.sparse import csr_matrix

        if init_method == "greedy_roi":
            res = _greedyROI(
                scan, num_components, soma_diameter, num_background_components
            )
            log("Refining initial components (HALS)...")
            res = initialization.hals(
                scan,
                res[0].reshape([image_height * image_width, -1], order="F"),
                res[1],
                res[2].reshape([image_height * image_width, -1], order="F"),
                res[3],
                maxIter=3,
            )
            initial_A, initial_C, initial_b, initial_f = res
        else:
            print(
                "Warning: Running sparse_nmf initialization on the entire field of view "
                "takes a lot of time."
            )
            res = initialization.initialize_components(
                scan,
                K=num_components,
                nb=num_background_components,
                method=init_method,
                alpha_snmf=snmf_alpha,
            )
            initial_A, initial_C, initial_b, initial_f, _ = res
        initial_A = csr_matrix(initial_A)
    log(initial_A.shape[-1], "components found...")

    # Remove bad components (based on spatial consistency and spiking activity)
    log("Removing bad components...")
    good_indices, _ = components_evaluation.estimate_components_quality(
        initial_C,
        mmap_scan,
        initial_A,
        initial_C,
        initial_b,
        initial_f,
        final_frate=fps,
        r_values_min=0.7,
        fitness_min=-20,
        fitness_delta_min=-20,
        dview=pool,
    )
    initial_A = initial_A[:, good_indices]
    initial_C = initial_C[good_indices]
    log(initial_A.shape[-1], "components remaining...")

    # Estimate noise per pixel
    log("Calculating noise per pixel...")
    pixels_noise, _ = pre_processing.get_noise_fft_parallel(
        mmap_scan, num_pixels_per_process, pool
    )

    # Update masks
    log("Updating masks...")
    A, b, C, f = spatial.update_spatial_components(
        mmap_scan,
        initial_C,
        initial_f,
        initial_A,
        b_in=initial_b,
        sn=pixels_noise,
        dims=(image_height, image_width),
        dview=pool,
        **params.get_group("spatial")
    )

    # Update traces (no impulse response modelling p=0)
    log("Updating traces...")
    res = temporal.update_temporal_components(
        mmap_scan,
        A,
        b,
        C,
        f,
        method="cvxpy",
        dview=pool,
        **params.get_group("temporal")
    )
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res
    R = YrA

    # Merge components
    log("Merging overlapping (and temporally correlated) masks...")
    merged_masks = ["dummy"]
    while len(merged_masks) > 0:
        res = merging.merge_components(
            mmap_scan,
            A,
            b,
            C,
            YrA,
            f,
            S,
            pixels_noise,
            params.get_group("temporal"),
            params.get_group("spatial"),
            dview=pool,
            thr=params.get("merging", "merge_thr"),
            bl=bl,
            c1=c1,
            sn=neurons_noise,
            g=g,
            mx=np.Inf,
            merge_parallel=params.get("merging", "merge_parallel"),
        )
        (
            A,
            C,
            num_components,
            merged_masks,
            S,
            bl,
            c1,
            neurons_noise,
            g,
            empty_merged,
            YrA,
        ) = res

    # Refine masks
    log("Refining masks...")
    A, b, C, f = spatial.update_spatial_components(
        mmap_scan,
        C,
        f,
        A,
        b_in=b,
        sn=pixels_noise,
        dims=(image_height, image_width),
        dview=pool,
        **params.get_group("spatial")
    )

    # Refine traces
    log("Refining traces...")
    res = temporal.update_temporal_components(
        mmap_scan,
        A,
        b,
        C,
        f,
        method="cvxpy",
        dview=pool,
        **params.get_group("temporal")
    )
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res
    R = YrA

    # Removing bad components (more stringent criteria)
    log("Removing bad components...")
    good_indices, _ = components_evaluation.estimate_components_quality(
        C + YrA,
        mmap_scan,
        A,
        C,
        b,
        f,
        final_frate=fps,
        r_values_min=0.8,
        fitness_min=-40,
        fitness_delta_min=-40,
        dview=pool,
    )
    A = A.toarray()[:, good_indices]
    C = C[good_indices]
    YrA = YrA[good_indices]
    log(A.shape[-1], "components remaining...")

    # Stop processes
    log("Done.")
    pool.close()

    # Get results
    masks = A.reshape(
        (image_height, image_width, -1), order="F"
    )  # h x w x num_components
    traces = C  # num_components x num_frames
    background_masks = b.reshape(
        (image_height, image_width, -1), order="F"
    )  # h x w x num_components
    background_traces = f  # num_background_components x num_frames
    raw_traces = C + YrA  # num_components x num_frames

    # Rescale traces to match scan range
    scaling_factor = np.sum(masks**2, axis=(0, 1)) / np.sum(masks, axis=(0, 1))
    traces = traces * np.expand_dims(scaling_factor, -1)
    raw_traces = raw_traces * np.expand_dims(scaling_factor, -1)
    masks = masks / scaling_factor
    background_scaling_factor = np.sum(background_masks**2, axis=(0, 1)) / np.sum(
        background_masks, axis=(0, 1)
    )
    background_traces = background_traces * np.expand_dims(
        background_scaling_factor, -1
    )
    background_masks = background_masks / background_scaling_factor

    return masks, traces, background_masks, background_traces, raw_traces

@profile
def extract_masks_new(
    scan,
    mmap_scan,
    num_components=200,
    num_background_components=1,
    merge_threshold=0.8,
    init_on_patches=True,
    init_method="greedy_roi",
    soma_diameter=(14, 14),
    snmf_alpha=0.5,
    patch_size=(50, 50),
    proportion_patch_overlap=0.2,
    num_components_per_patch=5,
    num_processes=8,
    num_pixels_per_process=5000,
    fps=15,
):
    # defined in Tolias lab pipeline
    num_components = 200
    num_background_components = 1
    merge_threshold = 0.8
    init_on_patches = True
    init_method = "greedy_roi"
    soma_diameter = (14, 14)
    snmf_alpha = 0.5
    patch_size = (50, 50)
    proportion_patch_overlap = 0.2
    num_components_per_patch = 5
    num_processes = 8
    num_pixels_per_process = 5000
    fps = 15
    p = 0
    ssub = 1
    tsub = max(int(fps / 2), 1)
    rolling_sum = True
    normalize_init = True
    rolling_length = 100
    block_size_temp = 10000
    check_nan = False
    method_deconvolution = "cvxpy"

    patch_size = np.array(patch_size)
    half_patch_size = np.int32(np.round(patch_size / 2))
    num_components_per_patch = int(round(num_components_per_patch))
    patch_overlap = np.int32(np.round(patch_size * proportion_patch_overlap))

    pool = mp.Pool(processes=num_processes)

    # all variables defined in CNMF parameters dictionary
    n_processes = num_processes  # default 8
    if init_on_patches:
        k = num_components_per_patch  # number of neurons per FOV
    else:
        k = num_components
    gSig = np.array(soma_diameter) / 2  # default [4,4]; expected half size of neurons
    gSiz = None  # default: [int(round((x * 2) + 1)) for x in gSig], half-size of bounding box for each neuron
    merge_thresh = (
        merge_threshold  # default 0.8; merging threshold, max correlation allowed
    )
    p = p  # default 2, order of the autoregressive process used to estimate deconvolution
    dview = pool  # default None
    Ain = None  # if known, it is the initial estimate of spatial filters
    Cin = None  # if knnown, initial estimate for calcium activity of each neuron
    b_in = None  # if known, initial estimate for background
    f_in = None  # if known, initial estimate of temporal profile of background activity
    do_merge = True  # Whether or not to merge
    ssub = ssub  # default 1; downsampleing factor in space
    tsub = tsub  # default 2; downsampling factor in time
    p_ssub = 1  # downsampling factor in space for patches
    p_tsub = 1  # downsampling factor in time for patches
    method_init = init_method  # default 'greedy_roi', can be greedy_roi or sparse_nmf
    alpha_snmf = snmf_alpha  # default 0.5, weight of the sparsity regularization
    rf = half_patch_size  # default None, half-size of the patches in pixels. rf=25, patches are 50x50
    stride = (
        patch_overlap  # default None, amount of overlap between the patches in pixels
    )
    memory_fact = 1  # unitless number accounting how much memory should be used. You will need to try different values to see which one would work; the default is OK for a 16 GB system
    gnb = num_background_components  # default 1; number of global background components
    nb_patch = num_background_components  # default 1; number of background components per patch
    only_init_patch = (
        init_on_patches  # default False; only run initialization on patches
    )
    method_deconvolution = method_deconvolution  # 'oasis' or 'cvxpy'; method used for deconvolution. Suggested 'oasis'
    n_pixels_per_process = num_pixels_per_process  # default 4000; Number of pixels to be processed in parallel per core (no patch mode). Decrease if memory problems
    block_size_temp = block_size_temp  # default 5000; Number of pixels to be used to perform residual computation in temporal blocks. Decrease if memory problems
    num_blocks_per_run_temp = 20  # In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing
    block_size_spat = 5000  # default 5000; Number of pixels to be used to perform residual computation in spatial blocks. Decrease if memory problems
    num_blocks_per_run_spat = 20  # In case of memory problems you can reduce this numbers, controlling the number of blocks processed in parallel during residual computing
    check_nan = check_nan  # Check if file contains NaNs (costly for very large files so could be turned off)
    skip_refinement = False  # Bool. If true it only performs one iteration of update spatial update temporal instead of two
    normalize_init = normalize_init  # Default True; Bool. Differences in intensities on the FOV might cause troubles in the initialization when patches are not used, so each pixels can be normalized by its median intensity
    options_local_NMF = None  # experimental, not to be used
    minibatch_shape = 100  # Number of frames stored in rolling buffer
    minibatch_suff_stat = 3  # mini batch size for updating sufficient statistics
    update_num_comps = True  # Whether to search for new components
    rval_thr = 0.9  # space correlation threshold for accepting a new component
    thresh_fitness_delta = -20  # Derivative test for detecting traces
    thresh_fitness_raw = None  # Threshold value for testing trace SNR
    thresh_overlap = 0.5  # Intersection-over-Union space overlap threshold for screening new components
    max_comp_update_shape = (
        np.inf
    )  # Maximum number of spatial components to be updated at each tim
    num_times_comp_updated = (
        np.inf
    )  # no description in documentation other than this is an int
    batch_update_suff_stat = (
        False  # Whether to update sufficient statistics in batch mode
    )
    s_min = None  # Minimum spike threshold amplitude (computed in the code if used).
    remove_very_bad_comps = False  # Bool (default False). whether to remove components with very low values of component quality directly on the patch. This might create some minor imprecisions.
    # However benefits can be considerable if done because if many components (>2000) are created and joined together, operation that causes a bottleneck
    border_pix = 0  # number of pixels to not consider in the borders
    low_rank_background = True  # if True the background is approximated with gnb components. If false every patch keeps its background (overlaps are randomly assigned to one spatial component only)
    # In the False case all the nonzero elements of the background components are updated using hals (to be used with one background per patch)
    update_background_components = (
        True  # whether to update the background components during the spatial phase
    )
    rolling_sum = rolling_sum  # default True; use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi
    rolling_length = (
        rolling_length  # default 100; width of rolling window for rolling sum option
    )
    min_corr = 0.85  # minimal correlation peak for 1-photon imaging initialization
    min_pnr = 20  # minimal peak  to noise ratio for 1-photon imaging initialization
    ring_size_factor = 1.5  # ratio between the ring radius and neuron diameters.
    center_psf = False  # whether to use 1p data processing mode. Set to true for 1p
    use_dense = True  # Whether to store and represent A and b as a dense matrix
    deconv_flag = True  # If True, deconvolution is also performed using OASIS
    simultaneously = False  # If true, demix and denoise/deconvolve simultaneously. Slower but can be more accurate.
    n_refit = 0  # Number of pools (cf. oasis.pyx) prior to the last one that are refitted when simultaneously demixing and denoising/deconvolving.
    del_duplicates = False  # whether to delete the duplicated created in initialization
    N_samples_exceptionality = None  # Number of consecutives intervals to be considered when testing new neuron candidates
    max_num_added = 3  # maximum number of components to be added at each step in OnACID
    min_num_trial = (
        2  # minimum numbers of attempts to include a new components in OnACID
    )
    thresh_CNN_noisy = (
        0.5  # threshold on the per patch CNN classifier for online algorithm
    )
    fr = fps  # default 30; imaging rate in frames per second
    decay_time = 0.4  # length of typical transient in seconds
    min_SNR = 2.5  # trace SNR threshold. Traces with SNR above this will get accepted
    ssub_B = 2  # downsampleing factor for 1-photon imaging background computation
    init_iter = 2  # number of iterations for 1-photon imaging initialization
    sniper_mode = False  # Whether to use the online CNN classifier for screening candidate components (otherwise space correlation is used)
    use_peak_max = False  # Whether to find candidate centroids using skimage's find local peaks function
    test_both = False  # Whether to use both the CNN and space correlation for screening new components
    expected_comps = (
        500  # number of expected components (for memory allocation purposes)
    )
    max_merge_area = None  # maximum area (in pixels) of merged components, used to determine whether to merge components during fitting process
    params = None  # specify params dictionary automatically instead of specifying all variables above

    if params is None:
        params = CNMFParams(
            border_pix=border_pix,
            del_duplicates=del_duplicates,
            low_rank_background=low_rank_background,
            memory_fact=memory_fact,
            n_processes=n_processes,
            nb_patch=nb_patch,
            only_init_patch=only_init_patch,
            p_ssub=p_ssub,
            p_tsub=p_tsub,
            remove_very_bad_comps=remove_very_bad_comps,
            rf=rf,
            stride=stride,
            check_nan=check_nan,
            n_pixels_per_process=n_pixels_per_process,
            k=k,
            center_psf=center_psf,
            gSig=gSig,
            gSiz=gSiz,
            init_iter=init_iter,
            method_init=method_init,
            min_corr=min_corr,
            min_pnr=min_pnr,
            gnb=gnb,
            normalize_init=normalize_init,
            options_local_NMF=options_local_NMF,
            ring_size_factor=ring_size_factor,
            rolling_length=rolling_length,
            rolling_sum=rolling_sum,
            ssub=ssub,
            ssub_B=ssub_B,
            tsub=tsub,
            block_size_spat=block_size_spat,
            num_blocks_per_run_spat=num_blocks_per_run_spat,
            block_size_temp=block_size_temp,
            num_blocks_per_run_temp=num_blocks_per_run_temp,
            update_background_components=update_background_components,
            method_deconvolution=method_deconvolution,
            p=p,
            s_min=s_min,
            do_merge=do_merge,
            merge_thresh=merge_thresh,
            decay_time=decay_time,
            fr=fr,
            min_SNR=min_SNR,
            rval_thr=rval_thr,
            N_samples_exceptionality=N_samples_exceptionality,
            batch_update_suff_stat=batch_update_suff_stat,
            expected_comps=expected_comps,
            max_comp_update_shape=max_comp_update_shape,
            max_num_added=max_num_added,
            min_num_trial=min_num_trial,
            minibatch_shape=minibatch_shape,
            minibatch_suff_stat=minibatch_suff_stat,
            n_refit=n_refit,
            num_times_comp_updated=num_times_comp_updated,
            simultaneously=simultaneously,
            sniper_mode=sniper_mode,
            test_both=test_both,
            thresh_CNN_noisy=thresh_CNN_noisy,
            thresh_fitness_delta=thresh_fitness_delta,
            thresh_fitness_raw=thresh_fitness_raw,
            thresh_overlap=thresh_overlap,
            update_num_comps=update_num_comps,
            use_dense=use_dense,
            use_peak_max=use_peak_max,
            alpha_snmf=alpha_snmf,
            max_merge_area=max_merge_area,
        )
    else:
        params = params
        params.set("patch", {"n_processes": n_processes})

    T = scan.shape[-1]
    params.set("online", {"init_batch": T})
    dims = scan.shape[:2]
    image_height, image_width = dims
    estimates = Estimates(A=Ain, C=Cin, b=b_in, f=f_in, dims=dims)

    # initialize on patches
    log("Initializing components...")
    (
        estimates.A,
        estimates.C,
        estimates.YrA,
        estimates.b,
        estimates.f,
        estimates.sn,
        estimates.optional_outputs,
    ) = map_reduce.run_CNMF_patches(
        mmap_scan.filename,
        dims + (T,),
        params,
        dview=dview,
        memory_fact=params.get("patch", "memory_fact"),
        gnb=params.get("init", "nb"),
        border_pix=params.get("patch", "border_pix"),
        low_rank_background=params.get("patch", "low_rank_background"),
        del_duplicates=params.get("patch", "del_duplicates"),
    )
    estimates.S = estimates.C

    estimates.bl, estimates.c1, estimates.g, estimates.neurons_sn = (
        None,
        None,
        None,
        None,
    )
    estimates.merged_ROIs = [0]

    # note: there are some if-else statements here that I skipped that may get run if params are set up differently

    # merge components
    while len(estimates.merged_ROIs) > 0:
        (
            estimates.A,
            estimates.C,
            estimates.nr,
            estimates.merged_ROIs,
            estimates.S,
            estimates.bl,
            estimates.c1,
            estimates.neurons_sn,
            estimates.g,
            empty_merged,
            estimates.YrA,
        ) = merging.merge_components(
            mmap_scan,
            estimates.A,
            estimates.b,
            estimates.C,
            estimates.YrA,
            estimates.f,
            estimates.S,
            estimates.sn,
            params.get_group("temporal"),
            params.get_group("spatial"),
            dview=dview,
            bl=estimates.bl,
            c1=estimates.c1,
            sn=estimates.neurons_sn,
            g=estimates.g,
            thr=params.get("merging", "merge_thr"),
            mx=np.Inf,
            fast_merge=True,
            merge_parallel=params.get("merging", "merge_parallel"),
            max_merge_area=None,
        )
        # max_merge_area=params.get('merging', 'max_merge_area'))

    log(estimates.A.shape[-1], "components found...")
    # Remove bad components (based on spatial consistency and spiking activity)
    log("Removing bad components...")
    good_indices, _ = components_evaluation.estimate_components_quality(
        estimates.C,
        mmap_scan,
        estimates.A,
        estimates.C,
        estimates.b,
        estimates.f,
        final_frate=fps,
        r_values_min=0.7,
        fitness_min=-20,
        fitness_delta_min=-20,
        dview=pool,
    )
    estimates.A = estimates.A[:, good_indices]
    estimates.C = estimates.C[good_indices]
    estimates.YrA = estimates.YrA[good_indices]
    estimates.S = estimates.S[good_indices]
    if estimates.bl is not None:
        estimates.bl = estimates.bl[good_indices]
    if estimates.c1 is not None:
        estimates.c1 = estimates.c1[good_indices]
    if estimates.neurons_sn is not None:
        estimates.neurons_sn = estimates.neurons_sn[good_indices]
    log(estimates.A.shape[-1], "components remaining...")

    # Update masks
    log("Updating masks...")
    (
        estimates.A,
        estimates.b,
        estimates.C,
        estimates.f,
    ) = spatial.update_spatial_components(
        mmap_scan,
        estimates.C,
        estimates.f,
        estimates.A,
        b_in=estimates.b,
        sn=estimates.sn,
        dims=dims,
        dview=pool,
        **params.get_group("spatial"),
    )

    # Update traces (no impulse response modelling p=0)
    log("Updating traces...")
    res = temporal.update_temporal_components(
        mmap_scan,
        estimates.A,
        estimates.b,
        estimates.C,
        estimates.f,
        method="cvxpy",
        dview=pool,
        **params.get_group("temporal"),
    )
    (
        estimates.C,
        estimates.A,
        estimates.b,
        estimates.f,
        estimates.S,
        estimates.bl,
        estimates.c1,
        estimates.neurons_sn,
        estimates.g,
        estimates.YrA,
        estimates.lam,
    ) = res

    # Merge components
    log("Merging overlapping (and temporally correlated) masks...")
    estimates.merged_ROIs = [0]
    # merge components
    while len(estimates.merged_ROIs) > 0:
        (
            estimates.A,
            estimates.C,
            estimates.nr,
            estimates.merged_ROIs,
            estimates.S,
            estimates.bl,
            estimates.c1,
            estimates.neurons_sn,
            estimates.g,
            empty_merged,
            estimates.YrA,
        ) = merging.merge_components(
            mmap_scan,
            estimates.A,
            estimates.b,
            estimates.C,
            estimates.YrA,
            estimates.f,
            estimates.S,
            estimates.sn,
            params.get_group("temporal"),
            params.get_group("spatial"),
            dview=dview,
            bl=estimates.bl,
            c1=estimates.c1,
            sn=estimates.neurons_sn,
            g=estimates.g,
            thr=params.get("merging", "merge_thr"),
            mx=np.Inf,
            fast_merge=True,
            merge_parallel=params.get("merging", "merge_parallel"),
            max_merge_area=None,
        )
        # max_merge_area=params.get('merging', 'max_merge_area'))

    # Refine masks
    log("Refining masks...")
    (
        estimates.A,
        estimates.b,
        estimates.C,
        estimates.f,
    ) = spatial.update_spatial_components(
        mmap_scan,
        estimates.C,
        estimates.f,
        estimates.A,
        b_in=estimates.b,
        sn=estimates.sn,
        dims=dims,
        dview=pool,
        **params.get_group("spatial"),
    )

    # Update traces (no impulse response modelling p=0)
    log("Refining traces...")
    res = temporal.update_temporal_components(
        mmap_scan,
        estimates.A,
        estimates.b,
        estimates.C,
        estimates.f,
        method="cvxpy",
        dview=pool,
        **params.get_group("temporal"),
    )
    (
        estimates.C,
        estimates.A,
        estimates.b,
        estimates.f,
        estimates.S,
        estimates.bl,
        estimates.c1,
        estimates.neurons_sn,
        estimates.g,
        estimates.YrA,
        estimates.lam,
    ) = res

    # Removing bad components (more stringent criteria)
    log("Removing bad components...")
    good_indices, _ = components_evaluation.estimate_components_quality(
        estimates.C + estimates.YrA,
        mmap_scan,
        estimates.A,
        estimates.C,
        estimates.b,
        estimates.f,
        final_frate=fps,
        r_values_min=0.8,
        fitness_min=-40,
        fitness_delta_min=-40,
        dview=pool,
    )
    estimates.A = estimates.A[:, good_indices]
    estimates.C = estimates.C[good_indices]
    estimates.YrA = estimates.YrA[good_indices]
    estimates.S = estimates.S[good_indices]
    if estimates.bl is not None:
        estimates.bl = estimates.bl[good_indices]
    if estimates.c1 is not None:
        estimates.c1 = estimates.c1[good_indices]
    if estimates.neurons_sn is not None:
        estimates.neurons_sn = estimates.neurons_sn[good_indices]
    log(estimates.A.shape[-1], "components remaining...")

    # Stop processes
    log("Done.")
    pool.close()

    estimates.normalize_components()

    # Get results
    masks = estimates.A.toarray().reshape(
        (image_height, image_width, -1), order="F"
    )  # h x w x num_components
    traces = estimates.C  # num_components x num_frames
    background_masks = estimates.b.reshape(
        (image_height, image_width, -1), order="F"
    )  # h x w x num_components
    background_traces = estimates.f  # num_background_components x num_frames
    raw_traces = estimates.C + estimates.YrA  # num_components x num_frames

    return masks, traces, background_masks, background_traces, raw_traces


def df_percentile(inputData, axis=None):
    """
    Extracting the percentile of the data where the mode occurs and its value.
    Used to determine the filtering level for DF/F extraction. Note that
    computation can be innacurate for short traces.
    """
    if axis is not None:

        def fnc(x):
            return df_percentile(x)

        result = np.apply_along_axis(fnc, axis, inputData)
        data_prct = result[:, 0]
        val = result[:, 1]
    else:
        # Create the function that we can use for the half-sample mode
        err = True
        while err:
            try:
                bandwidth, mesh, density, cdf = kde(inputData)
                err = False
            except:
                if type(inputData) is not list:
                    inputData = inputData.tolist()
                inputData += inputData

        data_prct = cdf[np.argmax(density)] * 100
        val = mesh[np.argmax(density)]
        if data_prct >= 100 or data_prct < 0:
            if type(inputData) is not list:
                inputData = inputData.tolist()
            inputData *= 2
            err = True
        if np.isnan(data_prct):
            data_prct = 50
            val = np.median(np.array(inputData))

    return data_prct, val

@profile
def deconvolve_detrended(trace, scan_fps, detrend_period=600, AR_order=2):
    """Same as the the `deconvolve` method, except that the fluorescence trace is detrended
    before autoregressive modeling

    :param np.array trace: 1-d array (num_frames) with the fluorescence trace.
    :param float scan_fps: fps of the scan
    :param float detrend_period: number of seconds over which percentiles are computed
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.

    :returns: Deconvolved spike trace.
    :returns: AR coefficients (AR_order) that model the calcium response:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...
    """

    detrend_window = int(round(detrend_period * scan_fps))
    n_chunks = len(trace) // detrend_window
    if detrend_window > 0 and n_chunks > 0:
        chunks_len = n_chunks * detrend_window
        trace_chunks = trace[:chunks_len].reshape(-1, detrend_window)
        data_prct = df_percentile(trace_chunks, axis=1)[0].mean()
        trace = trace - percentile_filter(trace, data_prct, detrend_window)

    _, _, _, AR_coeffs, _, spike_trace, _ = deconvolution.constrained_foopsi(
        trace, p=AR_order, method="cvxpy", bas_nonneg=False, fudge_factor=0.96
    )

    return spike_trace, AR_coeffs


def kde(data, N=None, MIN=None, MAX=None):
    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2 ** scipy.ceil(scipy.log2(N)))
    if MIN is None or MAX is None:
        minimum = min(data)
        maximum = max(data)
        Range = maximum - minimum
        MIN = minimum - Range / 10 if MIN is None else MIN
        MAX = maximum + Range / 10 if MAX is None else MAX

    # Range of the data
    R = MAX - MIN

    # Histogram the data to get a crude first approximation of the density
    M = len(data)
    DataHist, bins = scipy.histogram(data, bins=N, range=(MIN, MAX))
    DataHist = DataHist / M
    DCTData = fftpack.dct(DataHist, norm=None)

    I = [iN * iN for iN in range(1, N)]
    SqDCTData = (DCTData[1:] / 2) ** 2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError:
        print("Oops!")
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * scipy.exp(-scipy.arange(N) ** 2 * scipy.pi**2 * t_star / 2)
    # Inverse DCT to get density
    density = fftpack.idct(SmDCTData, norm=None) * N / R
    mesh = [(bins[i] + bins[i + 1]) / 2 for i in range(N)]
    bandwidth = scipy.sqrt(t_star) * R

    density = density / scipy.trapz(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])

    return bandwidth, mesh, density, cdf


def fixed_point(t, M, I, a2):
    l = 7
    I = scipy.float64(I)
    M = scipy.float64(M)
    a2 = scipy.float64(a2)
    f = (
        2
        * scipy.pi ** (2 * l)
        * scipy.sum(I**l * a2 * scipy.exp(-I * scipy.pi**2 * t))
    )
    for s in range(l, 1, -1):
        K0 = scipy.prod(range(1, 2 * s, 2)) / scipy.sqrt(2 * scipy.pi)
        const = (1 + (1 / 2) ** (s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
        f = (
            2
            * scipy.pi ** (2 * s)
            * scipy.sum(I**s * a2 * scipy.exp(-I * scipy.pi**2 * time))
        )
    return t - (2 * M * scipy.sqrt(scipy.pi) * f) ** (-2 / 5)


if __name__ == "__main__":
    pipeline_with_dataload("data/raster_motion_miniscan_export01.pkl")
