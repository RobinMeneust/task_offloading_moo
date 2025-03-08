"""This module contains utility functions for the task offloading MOO project."""

from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
import os
import numpy as np


def dominates(x, z):
    """Check if x dominates z on a minimization problem.

    Args:
        x (np.ndarray): First solution.
        z (np.ndarray): Second solution.

    Returns:
        bool: Whether x dominates z or not.
    """
    no_worse = np.all(x <= z)
    strictly_better = np.any(x < z)

    return no_worse and strictly_better


def save_generations_video_pymoo(history, out_path, file_name_without_extension):
    """Save the generations of a pymoo optimization history as a video.

    Args:
        history (list): List of pymoo optimization history objects.
        out_path (str): Path to save the video to.
        title (str): Title of the video.
    """
    os.makedirs(out_path, exist_ok=True)
    out_file_path = os.path.join(out_path, file_name_without_extension + ".mp4")

    with Recorder(Video(out_file_path)) as rec:

        # for each algorithm object in the history
        for entry in history:
            sc = Scatter(title=("Gen %s" % entry.n_gen))

            full_pop = entry.pop.get("F")
            opt_pop = entry.opt.get("F")
            full_pop = np.array([x for x in full_pop if x not in opt_pop])

            sc.add(full_pop, color="blue")
            sc.add(opt_pop, color="red")

            sc.do()

            # finally record the current visualization to the video
            rec.record()
