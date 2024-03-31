"""This file contains the pipeline that will be profiled by the memory_profiler package. This file implements the old version of CAIMAN (as opposed to pipeline_mprof_new.py implementing the new version of CAIMAN). Because memory_profiler requires decorators to help it run, this code cannot be imported to other files without error (unless you were to implement something that could ignore the decorators). Use pipeline_memray_old.py for that function (or better practice probably would be to make a new file that is more readable).
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
        ) = extract_masks(scan, mmap_scan, **params)

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

"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)."""
import numpy as np
import multiprocessing as mp
from caiman import components_evaluation
from caiman.utils import visualization
from caiman.source_extraction.cnmf import map_reduce, initialization, pre_processing, \
                                          merging, spatial, temporal, deconvolution
# from caiman_stats import df_percentile # comment out this line because caiman_stats file was added to the bottom of this file
from scipy.ndimage import percentile_filter
import glob, os, sys, time


def log(*messages):
    """ Simple logging function."""
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True, file=sys.__stdout__)


def mute_function(f):
    """ Decorator to ignore any standard output of the function."""
    def wrapper(*args, **kwargs):
        try:
            sys.stdout = open(os.devnull, 'w')
            return f(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__ # go back to normal (even after exceptions)
    return wrapper


@profile
def extract_masks(scan, mmap_scan, num_components=200, num_background_components=1,
                  merge_threshold=0.8, init_on_patches=True, init_method='greedy_roi',
                  soma_diameter=(14, 14), snmf_alpha=None, patch_size=(50, 50),
                  proportion_patch_overlap=0.2, num_components_per_patch=5,
                  num_processes=8, num_pixels_per_process=5000, fps=15):
    """ Extract masks from multi-photon scans using CNMF.

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
    print('Starting extract_masks')
    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Start processes
    log('Starting {} processes...'.format(num_processes))
    pool = mp.Pool(processes=num_processes)

    # Initialize components
    log('Initializing components...')
    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

        # Make sure they are integers
        patch_size = np.array(patch_size)
        half_patch_size = np.int32(np.round(patch_size / 2))
        num_components_per_patch = int(round(num_components_per_patch))
        patch_overlap = np.int32(np.round(patch_size * proportion_patch_overlap))

        # Create options dictionary (needed for run_CNMF_patches)
        options = {'patch_params': {'ssub': 'UNUSED.', 'tsub': 'UNUSED', 'nb': num_background_components,
                                    'only_init': True, 'skip_refinement': 'UNUSED.',
                                    'remove_very_bad_comps': False}, # remove_very_bads_comps unnecesary (same as default)
                   'preprocess_params': {'check_nan': False}, # check_nan is unnecessary (same as default value)
                   'spatial_params': {'nb': num_background_components}, # nb is unnecessary, it is pased to the function and in init_params
                   'temporal_params': {'p': 0, 'method': 'UNUSED.', 'block_size': 'UNUSED.'},
                   'init_params': {'K': num_components_per_patch, 'gSig': np.array(soma_diameter)/2,
                                   'gSiz': None, 'method': init_method, 'alpha_snmf': snmf_alpha,
                                   'nb': num_background_components, 'ssub': 1, 'tsub': max(int(fps / 2), 1),
                                   'options_local_NMF': 'UNUSED.', 'normalize_init': True,
                                   'rolling_sum': True, 'rolling_length': 100, 'min_corr': 'UNUSED',
                                   'min_pnr': 'UNUSED', 'deconvolve_options_init': 'UNUSED',
                                   'ring_size_factor': 'UNUSED', 'center_psf': 'UNUSED'},
                                   # gSiz, ssub, tsub, options_local_NMF, normalize_init, rolling_sum unnecessary (same as default values)
                   'merging' : {'thr': 'UNUSED.'}}

        # Initialize per patch
        res = map_reduce.run_CNMF_patches(mmap_scan.filename, (image_height, image_width, num_frames),
                                          options, rf=half_patch_size, stride=patch_overlap,
                                          gnb=num_background_components, dview=pool)
        initial_A, initial_C, YrA, initial_b, initial_f, pixels_noise, _ = res

        # Merge spatially overlapping components
        merged_masks = ['dummy']
        while len(merged_masks) > 0:
            res = merging.merge_components(mmap_scan, initial_A, initial_b, initial_C,
                                           initial_f, initial_C, pixels_noise,
                                           {'p': 0, 'method': 'cvxpy'}, spatial_params='UNUSED',
                                           dview=pool, thr=merge_threshold, mx=np.Inf)
            initial_A, initial_C, num_components, merged_masks, S, bl, c1, neurons_noise, g = res

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            try:
                os.remove(log_file)
            except FileNotFoundError:
                continue
                
    else:
        from scipy.sparse import csr_matrix
        if init_method == 'greedy_roi':
            res = _greedyROI(scan, num_components, soma_diameter, num_background_components)
            log('Refining initial components (HALS)...')
            res = initialization.hals(scan, res[0].reshape([image_height * image_width, -1], order='F'),
                                      res[1], res[2].reshape([image_height * image_width, -1], order='F'),
                                      res[3], maxIter=3)
            initial_A, initial_C, initial_b, initial_f = res
        else:
            print('Warning: Running sparse_nmf initialization on the entire field of view '
                  'takes a lot of time.')
            res = initialization.initialize_components(scan, K=num_components, nb=num_background_components,
                                                       method=init_method, alpha_snmf=snmf_alpha)
            initial_A, initial_C, initial_b, initial_f, _ = res
        initial_A = csr_matrix(initial_A)
    log(initial_A.shape[-1], 'components found...')

    # Remove bad components (based on spatial consistency and spiking activity)
    log('Removing bad components...')
    good_indices, _ = components_evaluation.estimate_components_quality(initial_C, scan,
        initial_A, initial_C, initial_b, initial_f, final_frate=fps, r_values_min=0.7,
        fitness_min=-20, fitness_delta_min=-20, dview=pool)
    initial_A = initial_A[:, good_indices]
    initial_C = initial_C[good_indices]
    log(initial_A.shape[-1], 'components remaining...')

    # Estimate noise per pixel
    log('Calculating noise per pixel...')
    pixels_noise, _ = pre_processing.get_noise_fft_parallel(mmap_scan, num_pixels_per_process, pool)

    # Update masks
    log('Updating masks...')
    A, b, C, f = spatial.update_spatial_components(mmap_scan, initial_C, initial_f, initial_A, b_in=initial_b,
                                                   sn=pixels_noise, dims=(image_height, image_width),
                                                   method='dilate', dview=pool,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components)

    # Update traces (no impulse response modelling p=0)
    log('Updating traces...')
    res = temporal.update_temporal_components(mmap_scan, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy', dview=pool)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res


    # Merge components
    log('Merging overlapping (and temporally correlated) masks...')
    merged_masks = ['dummy']
    while len(merged_masks) > 0:
        res = merging.merge_components(mmap_scan, A, b, C, f, S, pixels_noise, {'p': 0, 'method': 'cvxpy'},
                                       'UNUSED', dview=pool, thr=merge_threshold, bl=bl, c1=c1,
                                       sn=neurons_noise, g=g)
        A, C, num_components, merged_masks, S, bl, c1, neurons_noise, g = res

    # Refine masks
    log('Refining masks...')
    A, b, C, f = spatial.update_spatial_components(mmap_scan, C, f, A, b_in=b, sn=pixels_noise,
                                                   dims=(image_height, image_width),
                                                   method='dilate', dview=pool,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components)

    # Refine traces
    log('Refining traces...')
    res = temporal.update_temporal_components(mmap_scan, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy', dview=pool)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res

    # Removing bad components (more stringent criteria)
    log('Removing bad components...')
    good_indices, _ = components_evaluation.estimate_components_quality(C + YrA, scan, A,
        C, b, f, final_frate=fps, r_values_min=0.8, fitness_min=-40, fitness_delta_min=-40,
        dview=pool)
    A = A.toarray()[:, good_indices]
    C = C[good_indices]
    YrA = YrA[good_indices]
    log(A.shape[-1], 'components remaining...')

    # Stop processes
    log('Done.')
    pool.close()

    # Get results
    masks = A.reshape((image_height, image_width, -1), order='F') # h x w x num_components
    traces = C  # num_components x num_frames
    background_masks = b.reshape((image_height, image_width, -1), order='F') # h x w x num_components
    background_traces = f  # num_background_components x num_frames
    raw_traces = C + YrA  # num_components x num_frames

    # Rescale traces to match scan range
    scaling_factor = np.sum(masks**2, axis=(0, 1)) / np.sum(masks, axis=(0, 1))
    traces = traces * np.expand_dims(scaling_factor, -1)
    raw_traces = raw_traces * np.expand_dims(scaling_factor, -1)
    masks = masks / scaling_factor
    background_scaling_factor = np.sum(background_masks**2, axis=(0, 1)) / np.sum(background_masks,
                                                                                  axis=(0,1))
    background_traces = background_traces * np.expand_dims(background_scaling_factor, -1)
    background_masks = background_masks / background_scaling_factor

    return masks, traces, background_masks, background_traces, raw_traces


def _save_as_memmap(scan, base_name='caiman', chunk_size=5000):
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
    filename = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'.format(base_name, image_height,
                                                                    image_width, num_frames)

    # Create memory mapped file
    mmap_scan = np.memmap(filename, mode='w+', shape=(num_pixels, num_frames), dtype=np.float32)
    for i in range(0, num_frames, chunk_size):
        chunk = scan[..., i: i + chunk_size].reshape((num_pixels, -1), order='F')
        mmap_scan[:, i: i + chunk_size] = chunk
    mmap_scan.flush()

    return mmap_scan


def _greedyROI(scan, num_components=200, neuron_size=(11, 11),
               num_background_components=1):
    """ Initialize components by searching for gaussian shaped, highly active squares.
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
    gaussian_stddev = np.array(neuron_size) / 4 # entire neuron in four standard deviations
    gaussian_kernel = _gaussian2d(gaussian_stddev)

    # Create residual scan (scan minus background)
    residual_scan = scan - np.mean(scan, axis=(0, 1)) # image-wise brightness
    background = ndimage.gaussian_filter(np.mean(residual_scan, axis=-1), neuron_size)
    residual_scan -= np.expand_dims(background, -1)

    # Create components
    masks = np.zeros([image_height, image_width, num_components], dtype=np.float32)
    traces = np.zeros([num_components, num_frames], dtype=np.float32)
    mean_frame = np.mean(residual_scan, axis=-1)
    for i in range(num_components):

        # Get center of next component
        neuron_locations = ndimage.gaussian_filter(mean_frame, gaussian_stddev)
        y, x = np.unravel_index(np.argmax(neuron_locations), [image_height, image_width])

        # Compute initial trace (bit messy because of edges)
        half_kernel = np.fix(np.array(gaussian_kernel.shape) / 2).astype(np.int32)
        big_yslice = slice(max(y - half_kernel[0], 0), y + half_kernel[0] + 1)
        big_xslice = slice(max(x - half_kernel[1], 0), x + half_kernel[1] + 1)
        kernel_yslice = slice(max(0, half_kernel[0] - y),
                              None if image_height > y + half_kernel[0] else image_height - y - half_kernel[0] - 1)
        kernel_xslice = slice(max(0, half_kernel[1] - x),
                              None if image_width > x + half_kernel[1] else image_width - x - half_kernel[1] - 1)
        cropped_kernel = gaussian_kernel[kernel_yslice, kernel_xslice]
        trace = np.average(residual_scan[big_yslice, big_xslice].reshape(-1, num_frames),
                           weights=cropped_kernel.ravel(), axis=0)

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
    residual_scan += np.mean(scan, axis=(0, 1)) # add back overall brightness
    residual_scan += np.expand_dims(background, -1) # and background
    if num_background_components == 1:
        background_masks = np.expand_dims(np.mean(residual_scan, axis=-1), axis=-1)
        background_traces = np.expand_dims(np.mean(residual_scan, axis=(0, 1)), axis=0)
    else:
        from sklearn.decomposition import NMF
        print("Warning: Fitting more than one background component uses scikit-learn's "
              "NMF and may take some time.""")
        model = NMF(num_background_components, random_state=123, verbose=True)

        flat_masks = model.fit_transform(residual_scan.reshape(-1, num_frames))
        background_masks = flat_masks.reshape([image_height, image_width, -1])
        background_traces = model.components_

    return masks, traces, background_masks, background_traces


def _gaussian2d(stddev, truncate=4):
    """ Creates a 2-d gaussian kernel truncated at 4 standard deviations (8 in total).

    :param (float, float) stddev: Standard deviations in y and x.
    :param float truncate: Number of stddevs at each side of the kernel.

    ..note:: Kernel sizes will always be odd.
    """
    from matplotlib import mlab
    half_kernel = np.round(stddev * truncate) # kernel_size = 2 * half_kernel + 1
    y, x = np.meshgrid(np.arange(-half_kernel[0], half_kernel[0] + 1),
                       np.arange(-half_kernel[1], half_kernel[1] + 1))
    kernel = mlab.bivariate_normal(x, y, sigmay=stddev[0], sigmax=stddev[1])
    return kernel


# Based on caiman.source_extraction.cnmf.initialization.finetune()
def _rank1_NMF(scan, trace, num_iterations=5):
    num_frames = scan.shape[-1]
    for i in range(num_iterations):
        mask = np.maximum(np.dot(scan, trace), 0)
        mask  = mask * np.sum(mask) / np.sum(mask ** 2)
        trace = np.average(scan.reshape(-1, num_frames), weights=mask.ravel(), axis=0)
    return mask, trace


def deconvolve(trace, AR_order=2):
    """ Deconvolve traces using noise constrained deconvolution (Pnevmatikakis et al., 2016)

    :param np.array trace: 1-d array (num_frames) with the fluorescence trace.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.

    :returns: Deconvolved spike trace.
    :returns: AR coefficients (AR_order) that model the calcium response:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...
    """
    _, _, _, AR_coeffs, _, spike_trace, _ = deconvolution.constrained_foopsi(trace,
        p=AR_order, method='cvxpy', bas_nonneg=False, fudge_factor=0.96)
        # fudge_factor is a regularization term

    return spike_trace, AR_coeffs

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

    _, _, _, AR_coeffs, _, spike_trace, _ = deconvolution.constrained_foopsi(trace,
        p=AR_order, method='cvxpy', bas_nonneg=False, fudge_factor=0.96)

    return spike_trace, AR_coeffs


def get_centroids(masks):
    """ Calculate the centroids of each mask (calls caiman's plot_contours).

    :param np.array masks: Masks (image_height x image_width x num_components)

    :returns: Centroids (num_components x 2) in y, x pixels of each component.
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Get centroids
    fake_background = np.empty([image_height, image_width]) # needed for plot contours
    coordinates = visualization.plot_contours(masks, fake_background)
    import matplotlib.pyplot as plt; plt.close()
    centroids = np.array([coordinate['CoM'] for coordinate in coordinates])

    return centroids


def classify_masks(masks, soma_diameter=(12, 12)):
    """ Uses a convolutional network to predict the probability per mask of being a soma.

    :param np.array masks: Masks (image_height x image_width x num_components)

    :returns: Soma predictions (num_components).
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Prepare input
    from scipy.sparse import coo_matrix
    masks = coo_matrix(masks)
    soma_radius = np.int32(np.round(np.array(soma_diameter)/2))

    model_path = '/data/pipeline/python/pipeline/data/cnn_model'
    probs, _ = components_evaluation.evaluate_components_CNN(masks, (image_height, image_width),
                                                             soma_radius, model_name=model_path)

    return probs[:, 1]



















# Legacy: Used in preprocess.ExtractRaw
def demix_and_deconvolve_with_cnmf(scan, num_components=200, AR_order=2,
                                   merge_threshold=0.8, num_processes=20,
                                   num_pixels_per_process=5000, block_size=10000,
                                   num_background_components=4, init_method='greedy_roi',
                                   soma_radius=(5, 5), snmf_alpha=None,
                                   init_on_patches=False, patch_downsampling_factor=None,
                                   percentage_of_patch_overlap=None):
    """ Extract spike train activity from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find neurons/components
    (locations) and their fluorescence traces (activity) in a timeseries of images, and
    deconvolves them using an autoregressive model of the calcium impulse response
    function. See Pnevmatikakis et al., 2016 for details.

    Default values work alright for somatic images.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: An estimate of neurons/spatial components in the scan.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.
    :param int merge_threshold: Maximal temporal correlation allowed between activity of
        overlapping components before merging them.
    :param int num_processes: Number of processes to run in parallel. None for as many
        processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
        iteration.
    :param int block_size: 'number of pixels to process at the same time for dot product'
    :param int num_background_components:  Number of background components to use.
    :param string init_method: Initialization method for the components.
        'greedy_roi':Look for a gaussian-shaped patch, apply rank-1 NMF, store components,
            calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
        'local_nmf': ...
    :param (float, float) soma_radius: Estimated neuron radius (in pixels) in y and x.
        Used in'greedy_roi' initialization to define the size of the gaussian window.
    :param int snmf_alpha: Regularization parameter (alpha) for the sparse NMF (if used).
    :param bool init_on_patches: If True, run the initialization methods on small patches
        of the scan rather than on the whole image.
    :param int patch_downsampling_factor: Division to the image dimensions to obtain patch
        dimensions, e.g., if original size is 256 and factor is 10, patches will be 26x26
    :param int percentage_of_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%)

    :returns Location matrix (image_height x image_width x num_components). Inferred
        location of each component.
    :returns Activity matrix (num_components x num_frames). Inferred fluorescence traces
         (spike train convolved with the fitted impulse response function).
    :returns: Inferred location matrix for background components (image_height x
         image_width x num_background_components).
    :returns: Inferred activity matrix for background components (image_height x
        image_width x num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames) obtained from the
        scan minus activity from background and other components.
    :returns: Spike matrix (num_components x num_frames). Deconvolved spike activity.
    :returns: Autoregressive process coefficients (num_components x AR_order) used to
        model the calcium impulse response of each component:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...

    ..note:: Based on code provided by Andrea Giovanucci.
    ..note:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Computation- and memory-intensive for big scans.
    """
    import caiman
    from caiman.source_extraction.cnmf import cnmf

    # Save as memory mapped file in F order (that's how caiman wants it)
    mmap_filename = _save_as_memmap(scan, base_name='/tmp/caiman', order='F').filename

    # 'Load' scan
    mmap_scan, (image_height, image_width), num_frames = caiman.load_memmap(mmap_filename)
    images = np.reshape(mmap_scan.T, (num_frames, image_height, image_width), order='F')

    # Start the ipyparallel cluster
    client, direct_view, num_processes = caiman.cluster.setup_cluster(
        n_processes=num_processes)

    # Optionally, run the initialization method in small patches to initialize components
    initial_A = None
    initial_C = None
    initial_f = None
    if init_on_patches:
        # Calculate patch size (only square patches allowed)
        bigger_dimension = max(image_height, image_width)
        smaller_dimension = min(image_height, image_width)
        patch_size = bigger_dimension / patch_downsampling_factor
        patch_size = min(patch_size, smaller_dimension) # if bigger than small dimension

        # Calculate num_components_per_patch
        num_nonoverlapping_patches = (image_height/patch_size) * (image_width/patch_size)
        num_components_per_patch = num_components / num_nonoverlapping_patches
        num_components_per_patch = max(num_components_per_patch, 1) # at least 1

        # Calculate patch overlap in pixels
        overlap_in_pixels = patch_size * percentage_of_patch_overlap

        # Make sure they are integers
        patch_size = int(round(patch_size))
        num_components_per_patch = int(round(num_components_per_patch))
        overlap_in_pixels = int(round(overlap_in_pixels))

        # Run CNMF on patches (only for initialization, no impulse response modelling p=0)
        model = cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                          rf=int(round(patch_size / 2)), stride=overlap_in_pixels,
                          k=num_components_per_patch, merge_thresh=merge_threshold,
                          method_init=init_method, gSig=soma_radius,
                          alpha_snmf=snmf_alpha, gnb=num_background_components,
                          n_pixels_per_process=num_pixels_per_process,
                          block_size=block_size, check_nan=False, dview=direct_view,
                          method_deconvolution='cvxpy')
        model = model.fit(images)

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            try:
                os.remove(log_file)
            except FileNotFoundError:
                continue

        # Get results
        initial_A = model.A
        initial_C = model.C
        initial_f = model.f

    # Run CNMF
    model = cnmf.CNMF(num_processes, k=num_components, p=AR_order,
                      merge_thresh=merge_threshold, gnb=num_background_components,
                      method_init=init_method, gSig=soma_radius, alpha_snmf=snmf_alpha,
                      n_pixels_per_process=num_pixels_per_process, block_size=block_size,
                      check_nan=False, dview=direct_view, Ain=initial_A, Cin=initial_C,
                      f_in=initial_f, method_deconvolution='cvxpy')
    model = model.fit(images)

    # Get final results
    location_matrix = model.A  # pixels x num_components
    activity_matrix = model.C  # num_components x num_frames
    background_location_matrix = model.b  # pixels x num_background_components
    background_activity_matrix = model.f  # num_background_components x num_frames
    spikes = model.S  # num_components x num_frames, spike_ traces
    raw_traces = model.C + model.YrA  # num_components x num_frames
    AR_coefficients = model.g  # AR_order x num_components

    # Reshape spatial matrices to be image_height x image_width x num_frames
    new_shape = (image_height, image_width, -1)
    location_matrix = location_matrix.toarray().reshape(new_shape, order='F')
    background_location_matrix = background_location_matrix.reshape(new_shape, order='F')
    AR_coefficients = np.array(list(AR_coefficients))  # unwrapping it (num_components x 2)

    # Stop ipyparallel cluster
    client.close()
    caiman.stop_server()

    # Delete memory mapped scan
    os.remove(mmap_filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_coefficients)

"""
The CaImAn package installed in our pipeline Docker container is outdated and does not contain 
some of the newer functions in caiman/utils/stats.py.  So this file is copied over
from the CaImAn repository so that certain useful statistical  functions can be used by the pipeline.

https://github.com/flatironinstitute/CaImAn/blob/12fac5dff79ca7b4dcdbc0f5b3ce4658b7618948/caiman/utils/stats.py
"""

from builtins import range
from past.utils import old_div

import logging
import numpy as np
import scipy

try:
    import numba
except:
    pass

from scipy.linalg.lapack import dpotrf, dpotrs
from scipy import fftpack

#%%


def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x):
            return mode_robust_fast(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode


#%%


def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:

        def fnc(x):
            return mode_robust(x, dtype=dtype)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:

                wMin = np.inf
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode


#%%
#@numba.jit("void(f4[:])")


def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = old_div(data.size, 2) + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])


def compressive_nmf(A, L, R, r, X=None, Y=None, max_iter=100, ls=0):
    """Implements compressive NMF using an ADMM method as described in 
    Tepper and Shapiro, IEEE TSP 2015
    min_{U,V,X,Y} ||A - XY||_F^2 s.t. U = LX >= 0 and V = YR >=0
    """
    #r_ov = L.shape[1]
    m = L.shape[0]
    n = R.shape[1]
    U = np.random.rand(m, r)
    V = np.random.rand(r, n)
    Y = V.dot(R.T)
    Lam = np.zeros(U.shape)
    Phi = np.zeros(V.shape)
    l = 1
    f = 1
    x = 1
    I = np.eye(r)
    it = 0
    while it < max_iter:
        it += 1
        X = np.linalg.solve(Y.dot(Y.T) + l*I, Y.dot(A.T) + (l*U.T - Lam.T).dot(L)).T
        Y = np.linalg.solve(X.T.dot(X) + f*I, X.T.dot(A) + (f*V - Phi - ls).dot(R.T))
        LX = L.dot(X)
        U = LX + Lam/l
        U = np.where(U>0, U, 0)
        YR = Y.dot(R)
        V = YR + Phi/f
        V = np.where(V>0, V, 0)
        Lam += x*l*(LX - U)
        Phi += x*f*(YR - V)
        print(it)

    return X, Y

#%% kernel density estimation


def mode_robust_kde(inputData, axis=None):
    """
    Extracting the dataset of the mode using kernel density estimation
    """
    if axis is not None:

        def fnc(x):
            return mode_robust_kde(x)

        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        bandwidth, mesh, density, cdf = kde(inputData)
        dataMode = mesh[np.argamax(density)]

    return dataMode


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
                logging.warning('Percentile computation failed. Duplicating ' + 'and trying again.')
                if type(inputData) is not list:
                    inputData = inputData.tolist()
                inputData += inputData

        data_prct = cdf[np.argmax(density)] * 100
        val = mesh[np.argmax(density)]
        if data_prct >= 100 or data_prct < 0:
            logging.warning('Invalid percentile computed possibly due ' + 'short trace. Duplicating and recomuputing.')
            if type(inputData) is not list:
                inputData = inputData.tolist()
            inputData *= 2
            err = True
        if np.isnan(data_prct):
            logging.warning('NaN percentile computed. Reverting to median.')
            data_prct = 50
            val = np.median(np.array(inputData))

    return data_prct, val


"""
An implementation of the kde bandwidth selection method outlined in:
Z. I. Botev, J. F. Grotowski, and D. P. Kroese. Kernel density
estimation via diffusion. The Annals of Statistics, 38(5):2916-2957, 2010.
Based on the implementation in Matlab by Zdravko Botev.
Daniel B. Smith, PhD
Updated 1-23-2013
"""


def kde(data, N=None, MIN=None, MAX=None):

    # Parameters to set up the mesh on which to calculate
    N = 2**12 if N is None else int(2**scipy.ceil(scipy.log2(N)))
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
    SqDCTData = (DCTData[1:] / 2)**2

    # The fixed point calculation finds the bandwidth = t_star
    guess = 0.1
    try:
        t_star = scipy.optimize.brentq(fixed_point, 0, guess, args=(M, I, SqDCTData))
    except ValueError:
        print('Oops!')
        return None

    # Smooth the DCTransformed data using t_star
    SmDCTData = DCTData * scipy.exp(-scipy.arange(N)**2 * scipy.pi**2 * t_star / 2)
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
    f = 2 * scipy.pi**(2 * l) * scipy.sum(I**l * a2 * scipy.exp(-I * scipy.pi**2 * t))
    for s in range(l, 1, -1):
        K0 = scipy.prod(range(1, 2 * s, 2)) / scipy.sqrt(2 * scipy.pi)
        const = (1 + (1 / 2)**(s + 1 / 2)) / 3
        time = (2 * const * K0 / M / f)**(2 / (3 + 2 * s))
        f = 2 * scipy.pi**(2 * s) * scipy.sum(I**s * a2 * scipy.exp(-I * scipy.pi**2 * time))
    return t - (2 * M * scipy.sqrt(scipy.pi) * f)**(-2 / 5)


def csc_column_remove(A, ind):
    """ Removes specified columns for a scipy.sparse csc_matrix
    Args:
        A: scipy.sparse.csc_matrix
            Input matrix
        ind: iterable[int]
            list or np.array with columns to be removed
    """
    d1, d2 = A.shape
    if 'csc_matrix' not in str(type(A)):
        logging.warning("Original matrix not in csc_format. Converting it" + " anyway.")
        A = scipy.sparse.csc_matrix(A)
    indptr = A.indptr
    ind_diff = np.diff(A.indptr).tolist()
    ind_sort = sorted(ind, reverse=True)
    data_list = [A.data[indptr[i]:indptr[i + 1]] for i in range(d2)]
    indices_list = [A.indices[indptr[i]:indptr[i + 1]] for i in range(d2)]
    for i in ind_sort:
        del data_list[i]
        del indices_list[i]
        del ind_diff[i]
    indptr_final = np.cumsum([0] + ind_diff)
    data_final = [item for sublist in data_list for item in sublist]
    indices_final = [item for sublist in indices_list for item in sublist]
    A = scipy.sparse.csc_matrix((data_final, indices_final, indptr_final), shape=[d1, d2 - len(ind)])
    return A


def pd_solve(a, b):
    """ Fast matrix solve for positive definite matrix a"""
    L, info = dpotrf(a)
    if info == 0:
        return dpotrs(L, b)[0]
    else:
        return np.linalg.solve(a, b)


if __name__ == "__main__":
    pipeline_with_dataload("data/raster_motion_miniscan_export01.pkl")
