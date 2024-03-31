"""
This file contains several VERY similar methods for performing segmentation.
Most vary only on how methods are called. Others use the older version of CAIMAN, while some use the newer version.
This may lead to some dependency issues when trying to import in other files. I apologize for the inconvenience, 
but this was never meant to be very distributable, more just for quick testing.

This files is a replicate of the file in the soma-segmentation-exploration repo because it is referenced in some of the timing files in this repo.
"""

import sys, time, glob, os
import numpy as np
import multiprocessing as mp
from caiman.source_extraction.cnmf import (
    map_reduce,
    merging,
    initialization,
    pre_processing,
    spatial,
    temporal,
)
from caiman import components_evaluation
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams


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
    from scipy.stats import multivariate_normal

    half_kernel = np.round(stddev * truncate)  # kernel_size = 2 * half_kernel + 1
    y, x = np.meshgrid(
        np.arange(-half_kernel[0], half_kernel[0] + 1),
        np.arange(-half_kernel[1], half_kernel[1] + 1),
    )
    rv = multivariate_normal([0, 0], [[stddev[1], 0], [0, stddev[0]]])
    Z = rv.pdf(np.dstack((x, y)))
    return Z


# Based on caiman.source_extraction.cnmf.initialization.finetune()
def _rank1_NMF(scan, trace, num_iterations=5):
    num_frames = scan.shape[-1]
    for i in range(num_iterations):
        mask = np.maximum(np.dot(scan, trace), 0)
        mask = mask * np.sum(mask) / np.sum(mask**2)
        trace = np.average(scan.reshape(-1, num_frames), weights=mask.ravel(), axis=0)
    return mask, trace


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

def extract_masks_old(scan, mmap_scan, num_components=200, num_background_components=1, merge_threshold=0.8, init_on_patches=True, init_method='greedy_roi',
                  soma_diameter=(14, 14), snmf_alpha=0.5, patch_size=(50, 50),
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
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.params import CNMFParams

    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Start processes
    log("Starting {} processes...".format(num_processes))
    pool = mp.Pool(processes=num_processes)

    # Initialize components
    log("Initializing components...")
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
            "n_processes": num_processes,
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

    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

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
            try:
                os.remove(log_file)
            except FileNotFoundError:
                continue

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
            params.set("init", {"K": num_components})
            res = initialization.initialize_components(scan, **params.get_group("init"))
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
        **params.get_group("spatial"),
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
        **params.get_group("temporal"),
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
        **params.get_group("spatial"),
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
        **params.get_group("temporal"),
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


def extract_masks_adapted_prep(scan, num_processes):
    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Start processes
    log("Starting {} processes...".format(num_processes))
    pool = mp.Pool(processes=num_processes)

    return image_height, image_width, num_frames, pool


def init_on_patches_prep(
    patch_size,
    num_components_per_patch,
    proportion_patch_overlap,
    num_background_components,
    num_pixels_per_process,
    soma_diameter,
    init_method,
    snmf_alpha,
    fps,
    num_processes,
):
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
            "n_processes": num_processes,
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

    return params


def run_CNMF_patches(mmap_scan, image_height, image_width, num_frames, params, pool):
    # Initialize per patch
    return map_reduce.run_CNMF_patches(
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


def merge_overlapping_components_initial(
    mmap_scan,
    initial_A,
    initial_b,
    initial_C,
    YrA,
    initial_f,
    pixels_noise,
    params,
    pool,
):
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
        return res


def delete_log_files(search_param):
    # Delete log files (one per patch)
    log_files = glob.glob(search_param)
    for log_file in log_files:
        try:
            os.remove(log_file)
        except FileNotFoundError:
            continue


def remove_bad_components(
    mmap_scan, initial_A, initial_C, initial_b, initial_f, fps, pool
):
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
    return initial_A[:, good_indices], initial_C[good_indices]


def calculate_noise_per_pixel(mmap_scan, num_pixels_per_process, pool):
    return pre_processing.get_noise_fft_parallel(
        mmap_scan, num_pixels_per_process, pool
    )


def update_masks(
    mmap_scan,
    initial_C,
    initial_f,
    initial_A,
    initial_b,
    pixels_noise,
    image_height,
    image_width,
    pool,
    params,
):
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
        **params.get_group("spatial"),
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
        **params.get_group("temporal"),
    )
    return res


def merge_components_final(
    mmap_scan, A, b, C, YrA, f, S, pixels_noise, params, pool, bl, c1, neurons_noise, g
):
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
    return res


def remove_bad_components_final(C, YrA, mmap_scan, A, b, f, fps, pool):
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
    return A.toarray()[:, good_indices], C[good_indices], YrA[good_indices]


def refine_masks(
    mmap_scan, C, f, A, b, pixels_noise, image_height, image_width, pool, params
):
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
        **params.get_group("spatial"),
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
        **params.get_group("temporal"),
    )
    return res


def extract_masks_adapted_post(pool, image_height, image_width, A, b, f, C, YrA):
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


def extract_masks_adapted_time_profiling(
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
    image_height, image_width, num_frames, pool = extract_masks_adapted_prep(
        scan, num_processes
    )

    # Initialize components
    log("Initializing components...")

    params = init_on_patches_prep(
        patch_size,
        num_components_per_patch,
        proportion_patch_overlap,
        num_background_components,
        num_pixels_per_process,
        soma_diameter,
        init_method,
        snmf_alpha,
        fps,
        num_processes,
    )

    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

        # Initialize per patch
        (
            initial_A,
            initial_C,
            YrA,
            initial_b,
            initial_f,
            pixels_noise,
            _,
        ) = run_CNMF_patches(
            mmap_scan, image_height, image_width, num_frames, params, pool
        )

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
        ) = merge_overlapping_components_initial(
            mmap_scan,
            initial_A,
            initial_b,
            initial_C,
            YrA,
            initial_f,
            pixels_noise,
            params,
            pool,
        )

        delete_log_files("caiman*_LOG_*")

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
            params.set("init", {"K": num_components})
            res = initialization.initialize_components(
                scan,
                **params.get_group("init"),
            )
            initial_A, initial_C, initial_b, initial_f, _ = res
        initial_A = csr_matrix(initial_A)
    log(initial_A.shape[-1], "components found...")

    # Remove bad components (based on spatial consistency and spiking activity)
    log("Removing bad components...")
    initial_A, initial_C = remove_bad_components(
        mmap_scan, initial_A, initial_C, initial_b, initial_f, fps, pool
    )
    log(initial_A.shape[-1], "components remaining...")

    # Estimate noise per pixel
    log("Calculating noise per pixel...")
    pixels_noise, _ = calculate_noise_per_pixel(mmap_scan, num_pixels_per_process, pool)

    # Update masks
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = update_masks(
        mmap_scan,
        initial_C,
        initial_f,
        initial_A,
        initial_b,
        pixels_noise,
        image_height,
        image_width,
        pool,
        params,
    )

    # Merge components
    log("Merging overlapping (and temporally correlated) masks...")
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
    ) = merge_components_final(
        mmap_scan,
        A,
        b,
        C,
        YrA,
        f,
        S,
        pixels_noise,
        params,
        pool,
        bl,
        c1,
        neurons_noise,
        g,
    )

    # Refine masks
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = refine_masks(
        mmap_scan, C, f, A, b, pixels_noise, image_height, image_width, pool, params
    )

    # Removing bad components (more stringent criteria)
    log("Removing bad components...")
    A, C, YrA = remove_bad_components_final(C, YrA, mmap_scan, A, b, f, fps, pool)
    log(A.shape[-1], "components remaining...")

    # Stop processes
    log("Done.")
    (
        masks,
        traces,
        background_masks,
        background_traces,
        raw_traces,
    ) = extract_masks_adapted_post(pool, image_height, image_width, A, b, f, C, YrA)

    return masks, traces, background_masks, background_traces, raw_traces


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
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.params import CNMFParams

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

def extract_masks_new_prep(scan):
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

    return estimates, params, pool, image_height, image_width, dims, T

def initialize_patches(estimates, mmap_scan, dims, params, T, dview):
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

    return estimates

def merging_initial(estimates, params, mmap_scan, dview):
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

    return estimates

def keep_good_components_initial(estimates, mmap_scan, fps, pool):
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

    return estimates

def update_spatial_temporal_initial(estimates, dims, pool, params, mmap_scan):
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

    return estimates

def merging_final(estimates, params, mmap_scan, dview):
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

    return estimates

def update_spatial_temporal_final(estimates, dims, pool, params, mmap_scan):

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

    return estimates

def keep_good_components_final(estimates, mmap_scan, fps, pool):

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

    return estimates

def extract_masks_new_post(estimates, pool, image_width, image_height):
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


def extract_masks_new_timing(
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
    
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.params import CNMFParams
    
    estimates, params, pool, image_height, image_width, dims, T = extract_masks_new_prep(scan)
    dview = pool
    

    # initialize on patches
    estimates = initialize_patches(estimates, mmap_scan, dims, params, T, dview)

    estimates = merging_initial(estimates, params, mmap_scan, dview)

    estimates = keep_good_components_initial(estimates, mmap_scan, fps, pool)

    estimates = update_spatial_temporal_initial(estimates, dims, pool, params, mmap_scan)

    estimates = merging_final(estimates, params, mmap_scan, dview)

    estimates = update_spatial_temporal_final(estimates, dims, pool, params, mmap_scan)

    estimates = keep_good_components_final(estimates, mmap_scan, fps, pool)

    masks, traces, background_masks, background_traces, raw_traces = extract_masks_new_post(estimates, pool, image_width, image_height)
    
    return masks, traces, background_masks, background_traces, raw_traces
