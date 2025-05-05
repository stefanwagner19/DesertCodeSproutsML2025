import os
import sys

sys.path.extend([os.path.join(os.path.dirname(__file__), ".")])

import pacman

sys.path.extend([os.path.join(os.path.dirname(__file__), "..")])

import mysettings


def run():
    rewards = mysettings.PacmanRewards()
    N_runs = mysettings.normalRuns
    N_training = mysettings.trainingRuns
    grid = mysettings.grid
    args = pacman.readCommand(
        [
            "-p",
            "PacmanQAgent",
            f"--numGames={N_runs + N_training}",
            f"--layout={grid}",
            f"--numTraining={N_training}",
        ]
    )
    args["rewardfn"] = rewards
    pacman.runGames(**args)
