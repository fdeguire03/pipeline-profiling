import cProfile
import pstats
import io
import os
from shutil import move
import matplotlib.pyplot as plt
from numpy import std
import numpy as np


def average(stats, count):
    stats.total_calls /= count
    stats.prim_calls /= count
    stats.total_tt /= count

    for func, source in stats.stats.items():
        cc, nc, tt, ct, callers = source
        stats.stats[func] = (cc / count, nc / count, tt / count, ct / count, callers)

    return stats


def average_profilings(target_profile_function, count, filename="test.txt", save_seg_results=False, input_index=False):
    output_stream = io.StringIO()
    profiler_status = pstats.Stats(stream=output_stream)
    individual_times = []

    for index in range(count):
        if input_index:
            start_time = profiler_status.total_tt
            profiler = cProfile.Profile()
            profiler.enable()
    
            output = target_profile_function(index)
    
            profiler.disable()
            profiler_status.add(profiler)
            end_time = profiler_status.total_tt
        else:
            start_time = profiler_status.total_tt
            profiler = cProfile.Profile()
            profiler.enable()
    
            output = target_profile_function()
    
            profiler.disable()
            profiler_status.add(profiler)
            end_time = profiler_status.total_tt

        print(
            f"Profiling #{index+1}: Function call took {round(end_time-start_time, 3)} seconds (total of {round(profiler_status.total_tt, 3)} seconds)."
        )
        individual_times.append(end_time - start_time)

        if save_seg_results:
            masks,traces,background_masks,background_traces,raw_traces = output
            np.save(f'{save_seg_results}/masks{index}.npy', masks)
            np.save(f'{save_seg_results}/traces{index}.npy', traces)
            np.save(f'{save_seg_results}/raw_traces{index}.npy', raw_traces)

    average(profiler_status, count)
    profiler_status.dump_stats(filename)
    return profiler_status, individual_times, output


def sort_files():
    filenames = os.listdir()
    for file in filenames:
        if os.path.isdir(file):
            continue
        if file.endswith("prof"):
            move(file, f"data/{file}")
            print(f"Moved {file} to data directory")
        elif file.endswith("dat"):
            move(file, f"memory_profiler_files/{file}")
            print(f"Moved {file} to memory_profiler_files directory")
        elif file.startswith("memray"):
            move(file, f"memray_files/{file}")
            print(f"Moved {file} to memray_files directory")
        else:
            print(f"Not moving {file}")
    print("Finished moving files")


def plot_profiling_times(individual_times, total_time, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(range(len(individual_times)), individual_times)
    ax.set_ylabel("seconds")
    ax.set_xlabel("Profile #")
    ax.set_title("Program run times during profiling")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    stdev = std(individual_times)
    ax.plot(range(len(individual_times)), len(individual_times) * [total_time], "--")
    ax.plot(
        range(len(individual_times)),
        len(individual_times) * [total_time + stdev],
        "k--",
    )
    ax.plot(
        range(len(individual_times)),
        len(individual_times) * [total_time - stdev],
        "k--",
    )

    return ax


def plot_profiling_hist(individual_times, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(individual_times)
    ax.set_xlabel("seconds")
    ax.set_ylabel("Count")
    ax.set_title("Program run times during profiling")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def plot_profiling_params(individual_times, total_time):
    fig, axes = plt.subplots(2, 1)
    plot_profiling_times(individual_times, total_time, axes[0])
    plot_profiling_hist(individual_times, axes[1])
    plt.tight_layout()
    plt.savefig(f"data/average_profile_{len(individual_times)}runs.png")
    return fig
