"""Custom archive implementation for MOFDA algorithm."""

import numpy as np
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MOFDAArchive:
    """Custom archive implementation for MOFDA algorithm."""

    def __init__(self, size=None):
        """Initialize the archive with a given size."""
        self.size = size
        self.data = Population()
        self.grid = None

    def __len__(self):
        """Return the number of solutions in the archive."""
        return len(self.data)

    def add(self, pop):
        """
        Add solutions to the archive and return the archive itself.

        This method is needed for compatibility with pymoo's algorithm framework.
        """
        self.update(pop)
        return self

    def get(self, key):
        """Get a solution from the archive by its key."""
        return self.data.get(key) if len(self.data) > 0 else None

    def _accept(self, pop):
        """Accept only non-dominated solutions to the archive."""
        # Safety check for empty populations
        if pop is None or len(pop) == 0:
            return np.array([], dtype=bool)

        # Get the population's objective values
        F_pop = pop.get("F")

        # If archive is empty, just do non-dominated sorting within the population
        if len(self.data) == 0:
            if len(pop) <= 1:
                return np.full(len(pop), True)
            else:
                # Find non-dominated solutions in the population
                fronts = NonDominatedSorting().do(F_pop, only_non_dominated_front=True)
                # Create acceptance mask
                accepted = np.zeros(len(pop), dtype=bool)

                # Handle both single integer and array cases
                if isinstance(fronts[0], (int, np.integer)):
                    accepted[fronts[0]] = True
                else:
                    for idx in fronts[0]:
                        accepted[idx] = True
                return accepted

        # If we have archive data, compare with current population
        F_archive = self.data.get("F")

        # Combine archive and population objectives for sorting
        F_all = np.vstack([F_archive, F_pop])

        # Find non-dominated solutions across both sets
        fronts = NonDominatedSorting().do(F_all, only_non_dominated_front=True)

        # Create acceptance mask for population solutions
        accepted = np.zeros(len(pop), dtype=bool)
        archive_size = len(F_archive)

        # Handle both single integer and array cases
        if isinstance(fronts[0], (int, np.integer)):
            if fronts[0] >= archive_size:
                accepted[fronts[0] - archive_size] = True
        else:
            for idx in fronts[0]:
                if idx >= archive_size:
                    accepted[idx - archive_size] = True

        return accepted

    def update(self, pop):
        """Update the archive with new solutions."""
        if pop is None or len(pop) == 0:
            return

        accept = self._accept(pop)
        if np.any(accept):
            accepted = pop[accept]
            if len(self.data) == 0:
                self.data = accepted
            else:
                self.data = Population.merge(self.data, accepted)
            self._filter()

    def _filter(self):
        if self.size is None or len(self.data) <= self.size:
            return

        # Use grid-based crowding distance
        F = self.data.get("F")
        n_dim = F.shape[1]
        n_grid = max(2, int(np.ceil(self.size ** (1 / n_dim))))

        grid_indices = np.zeros((len(F), n_dim), dtype=int)
        for i in range(n_dim):
            f_min, f_max = F[:, i].min(), F[:, i].max()
            delta = (f_max - f_min) / n_grid if f_max > f_min else 1.0
            grid_indices[:, i] = np.minimum(np.floor((F[:, i] - f_min) / delta), n_grid - 1)

        # Count solutions in each cell
        unique_cells, cell_counts = np.unique(grid_indices, axis=0, return_counts=True)
        crowding = np.zeros(len(F))
        for i, indices in enumerate(grid_indices):
            idx = np.where((unique_cells == indices).all(axis=1))[0][0]
            crowding[i] = cell_counts[idx]

        # Keep the least crowded solutions
        least_crowded_solutions = np.argsort(crowding)[: self.size]
        self.data = self.data[least_crowded_solutions]

        # Update grid information
        self.grid = {"indices": grid_indices[least_crowded_solutions], "counts": crowding[least_crowded_solutions]}

    def get_leader(self):
        """Get the leader solution from the archive."""
        if len(self.data) == 0:
            return None

        if self.grid is None:
            return self.data[np.random.randint(len(self.data))]

        # Select from less crowded regions
        probs = 1.0 / self.grid["counts"]
        probs = probs / np.sum(probs)
        idx = np.random.choice(len(self.data), p=probs)
        return self.data[idx]
