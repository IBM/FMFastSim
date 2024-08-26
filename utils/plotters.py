#
# Copyright (c) 2024 by Contributors for FMFastSim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from dataloader.data_handler import DataInfo
from utils.observables import LongitudinalProfile, ProfileType, Profile, Energy, LateralProfile, PhiProfile

plt.rcParams.update({"font.size": 22})

FULL_SIM_HISTOGRAM_COLOR = "blue"
FULL_SIM_GAUSSIAN_COLOR = "green"
ML_SIM_HISTOGRAM_COLOR = "red"
ML_SIM_GAUSSIAN_COLOR = "orange"
HISTOGRAM_TYPE = "step"

@dataclass
class Plotter:
    """ An abstract class defining interface of all plotters.

    Do not use this class directly. Use ProfilePlotter or EnergyPlotter instead.

    Attributes:
        _save_dir: directory to save the figures
        _particle_energy: An integer which is energy of the primary particle in GeV units.
        _particle_theta: An integer which is an angle of the primary particle in degrees.
        _geometry: A string which is a name of the calorimeter geometry (e.g. SiW, SciPb).

    """
    _save_dir: str
    _particle_energy: float
    _particle_theta: float
    _geometry: str

    def plot_and_save(self):
        pass


def _gaussian(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """ Computes a value of a Gaussian.

    Args:
        x: An argument of a function.
        a: A scaling parameter.
        mu: A mean.
        sigma: A variance.

    Returns:
        A value of a function for given arguments.

    """
    return a * np.exp(-((x - mu)**2 / (2 * sigma**2)))


def _best_fit(data: np.ndarray,
              bins: np.ndarray,
              hist: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """ Finds estimated shape of a Gaussian using Use non-linear least squares.

    Args:
        data: A numpy array with values of observables from multiple events.
        bins: A numpy array specifying histogram bins.
        hist: If histogram is calculated. Then data is the frequencies.

    Returns:
        A tuple of two lists. Xs and Ys of predicted curve.

    """
    # Calculate histogram.
    if not hist:
        hist, _ = np.histogram(data, bins)
    else:
        hist = data

    # Choose only those bins which are nonzero. Nonzero() return a tuple of arrays. In this case it has a length = 1,
    # hence we are interested in its first element.
    indices = hist.nonzero()[0]

    # Based on previously chosen nonzero bin, calculate position of xs and ys_bar (true values) which will be used in
    # fitting procedure. Len(bins) == len(hist + 1), so we choose middles of bins as xs.
    bins_middles = (bins[:-1] + bins[1:]) / 2
    xs = bins_middles[indices]
    ys_bar = hist[indices]

    # Set initial parameters for curve fitter.
    a0 = np.max(ys_bar)
    mu0 = np.mean(xs)
    sigma0 = np.var(xs)

    # Fit a Gaussian to the prepared data.
    (a, mu, sigma), _ = curve_fit(f=_gaussian,
                                  xdata=xs,
                                  ydata=ys_bar,
                                  p0=[a0, mu0, sigma0],
                                  method="trf",
                                  maxfev=1000)

    # Calculate values of an approximation in given points and return values.
    ys = _gaussian(xs, a, mu, sigma)
    return xs, ys


@dataclass
class ProfilePlotter(Plotter):
    """ Plotter responsible for preparing plots of profiles and their first and second moments.

    Attributes:
        _full_simulation: A numpy array representing a profile of data generated by Geant4.
        _ml_simulation: A numpy array representing a profile of data generated by ML model.
        _plot_gaussian: A boolean. Decides whether first and second moment should be plotted as a histogram or
            a fitted gaussian.
        _profile_type: An enum. A profile can be either lateral or longitudinal.

    """
    _full_simulation: Profile
    _ml_simulation: Profile
    _plot_gaussian: bool = False
    _particle_phi: float = None

    def __post_init__(self):
        # Check if profiles are either both longitudinal or lateral.
        full_simulation_type = type(self._full_simulation)
        ml_generation_type = type(self._ml_simulation)
        assert full_simulation_type == ml_generation_type, "Both profiles within a ProfilePlotter must be the same " \
                                                           "type."

        # Set an attribute with profile type.
        if full_simulation_type == LongitudinalProfile:
            self._profile_type = ProfileType.LONGITUDINAL
        elif full_simulation_type == LateralProfile:
            self._profile_type = ProfileType.LATERAL
        elif full_simulation_type == PhiProfile:
            self._profile_type = ProfileType.PHIPROFILE

    def _plot_and_save_customizable_histogram(
            self,
            full_simulation: np.ndarray,
            ml_simulation: np.ndarray,
            bins: np.ndarray,
            xlabel: str,
            observable_name: str,
            plot_profile: bool = False,
            y_log_scale: bool = False) -> None:
        """ Prepares and saves a histogram for a given pair of observables.

        Args:
            full_simulation: A numpy array of observables coming from full simulation.
            ml_simulation: A numpy array of observables coming from ML simulation.
            bins: A numpy array specifying histogram bins.
            xlabel: A string. Name of x-axis on the plot.
            observable_name: A string. Name of plotted observable.
            plot_profile: A boolean. If set to True, full_simulation and ml_simulation are histogram weights while x is
                defined by the number of layers. This means that in order to plot histogram (and gaussian), one first
                need to create a data repeating each layer or R index appropriate number of times. Should be set to True
                only while plotting profiles not first or second moments.
            y_log_scale: A boolean. Used log scale on y-axis is set to True.

        Returns:
            None.

        """
        fig, axes = plt.subplots(2,
                                 1,
                                 figsize=(15, 10),
                                 clear=True,
                                 sharex="all")

        # Plot histograms.
        if plot_profile:
            # We already have the bins (layers) and freqencies (energies),
            # therefore directly plotting a step plot + lines instead of a hist plot.
            axes[0].step(bins[:-1],
                         full_simulation,
                         label="FullSim",
                         color=FULL_SIM_HISTOGRAM_COLOR,
                         where='post')
            axes[0].step(bins[:-1],
                         ml_simulation,
                         label="MLSim",
                         color=ML_SIM_HISTOGRAM_COLOR,
                         where='post')
            axes[0].vlines(x=bins[0],
                           ymin=0,
                           ymax=full_simulation[0],
                           color=FULL_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[-2],
                           ymin=0,
                           ymax=full_simulation[-1],
                           color=FULL_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[0],
                           ymin=0,
                           ymax=ml_simulation[0],
                           color=ML_SIM_HISTOGRAM_COLOR)
            axes[0].vlines(x=bins[-2],
                           ymin=0,
                           ymax=ml_simulation[-1],
                           color=ML_SIM_HISTOGRAM_COLOR)
            axes[0].set_ylim(0, None)

            # For using it later for the ratios.
            energy_full_sim, energy_ml_sim = full_simulation, ml_simulation
        else:
            energy_full_sim, _, _ = axes[0].hist(
                x=full_simulation,
                bins=bins,
                label="FullSim",
                histtype=HISTOGRAM_TYPE,
                color=FULL_SIM_HISTOGRAM_COLOR)
            energy_ml_sim, _, _ = axes[0].hist(x=ml_simulation,
                                               bins=bins,
                                               label="MLSim",
                                               histtype=HISTOGRAM_TYPE,
                                               color=ML_SIM_HISTOGRAM_COLOR)

        # Plot Gaussians if needed.
        if self._plot_gaussian:
            if plot_profile:
                (xs_full_sim, ys_full_sim) = _best_fit(full_simulation,
                                                       bins,
                                                       hist=True)
                (xs_ml_sim, ys_ml_sim) = _best_fit(ml_simulation,
                                                   bins,
                                                   hist=True)
            else:
                (xs_full_sim, ys_full_sim) = _best_fit(full_simulation, bins)
                (xs_ml_sim, ys_ml_sim) = _best_fit(ml_simulation, bins)
            axes[0].plot(xs_full_sim,
                         ys_full_sim,
                         color=FULL_SIM_GAUSSIAN_COLOR,
                         label="FullSim")
            axes[0].plot(xs_ml_sim,
                         ys_ml_sim,
                         color=ML_SIM_GAUSSIAN_COLOR,
                         label="MLSim")

        if y_log_scale:
            axes[0].set_yscale("log")
        axes[0].legend(loc="best")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel("Energy [Mev]")
        axes[0].set_title(
            f" $e^-$, {self._particle_energy} [GeV], {self._particle_theta}$^{{\circ}}$, {self._geometry}"
        )

        # Calculate ratios.
        ratio = np.divide(energy_ml_sim,
                          energy_full_sim,
                          out=np.ones_like(energy_ml_sim),
                          where=(energy_full_sim != 0))
        # Since len(bins) == 1 + data, we calculate middles of bins as xs.
        bins_middles = (bins[:-1] + bins[1:]) / 2
        axes[1].plot(bins_middles, ratio, "-o")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("MLSim/FullSim")
        axes[1].axhline(y=1, color="black")
        plt.savefig(
            f"{self._save_dir}/{observable_name}_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png"))
        plt.clf()

    def _plot_profile(self) -> None:
        """ Plots profile of an observable.

        Returns:
            None.

        """
        full_simulation_profile = self._full_simulation.calc_profile()
        ml_simulation_profile = self._ml_simulation.calc_profile()
        if self._profile_type == ProfileType.LONGITUDINAL:
            # matplotlib will include the right-limit for the last bar,
            # hence extending by 1.
            bins = np.linspace(0, DataInfo().N_CELLS_Z, DataInfo().N_CELLS_Z + 1)
            observable_name = "LongProf"
            xlabel = "Layer index"
        elif self._profile_type == ProfileType.LATERAL:
            bins = np.linspace(0, DataInfo().N_CELLS_R, DataInfo().N_CELLS_R + 1)
            observable_name = "LatProf"
            xlabel = "R index"
        elif self._profile_type == ProfileType.PHIPROFILE:
            bins = np.linspace(0, DataInfo().N_CELLS_PHI, DataInfo().N_CELLS_PHI + 1)
            observable_name = "PhiProf"
            xlabel = "Phi index"
        self._plot_and_save_customizable_histogram(full_simulation_profile,
                                                   ml_simulation_profile,
                                                   bins,
                                                   xlabel,
                                                   observable_name,
                                                   plot_profile=True)

    def _plot_first_moment(self) -> None:
        """ Plots and saves a first moment of an observable's profile.

        Returns:
            None.

        """
        full_simulation_first_moment = self._full_simulation.calc_first_moment(
        )
        ml_simulation_first_moment = self._ml_simulation.calc_first_moment()
        if self._profile_type == ProfileType.LONGITUDINAL:
            xlabel = "$<\lambda> [mm]$"
            observable_name = "LongFirstMoment"
            bins = np.linspace(0, 0.4 * DataInfo().N_CELLS_Z * DataInfo().SIZE_Z, 128)
        else:
            xlabel = "$<r> [mm]$"
            observable_name = "LatFirstMoment"
            bins = np.linspace(0, 0.75 * DataInfo().N_CELLS_R * DataInfo().SIZE_R, 128)

        self._plot_and_save_customizable_histogram(
            full_simulation_first_moment, ml_simulation_first_moment, bins,
            xlabel, observable_name)

    def _plot_second_moment(self) -> None:
        """ Plots and saves a second moment of an observable's profile.

        Returns:
            None.

        """
        full_simulation_second_moment = self._full_simulation.calc_second_moment(
        )
        ml_simulation_second_moment = self._ml_simulation.calc_second_moment()
        if self._profile_type == ProfileType.LONGITUDINAL:
            xlabel = "$<\lambda^{2}> [mm^{2}]$"
            observable_name = "LongSecondMoment"
            bins = np.linspace(0, pow(DataInfo().N_CELLS_Z * DataInfo().SIZE_Z, 2) / 35., 128)
        else:
            xlabel = "$<r^{2}> [mm^{2}]$"
            observable_name = "LatSecondMoment"
            bins = np.linspace(0, pow(DataInfo().N_CELLS_R * DataInfo().SIZE_R, 2) / 8., 128)

        self._plot_and_save_customizable_histogram(
            full_simulation_second_moment, ml_simulation_second_moment, bins,
            xlabel, observable_name)

    def plot_and_save(self) -> None:
        """ Main plotting function.

        Calls private methods and prints the information about progress.

        Returns:
            None.

        """
        if self._profile_type == ProfileType.LONGITUDINAL:
            profile_type_name = "longitudinal"
        elif self._profile_type == ProfileType.LATERAL:
            profile_type_name = "lateral"
        elif self._profile_type == ProfileType.PHIPROFILE:
            profile_type_name = "phi_profile"
        print(f"Plotting the {profile_type_name} profile...")
        self._plot_profile()
        if not self._profile_type == ProfileType.PHIPROFILE:
            print(f"Plotting the first moment of {profile_type_name} profile...")
            self._plot_first_moment()
            print(f"Plotting the second moment of {profile_type_name} profile...")
            self._plot_second_moment()


@dataclass
class EnergyPlotter(Plotter):
    """ Plotter responsible for preparing plots of profiles and their first and second moments.

    Attributes:
        _full_simulation: A numpy array representing a profile of data generated by Geant4.
        _ml_simulation: A numpy array representing a profile of data generated by ML model.

    """
    _full_simulation: Energy
    _ml_simulation: Energy
    _particle_phi: float = None

    def _plot_total_energy(self, y_log_scale=True) -> None:
        """ Plots and saves a histogram with total energy detected in an event.

        Args:
            y_log_scale: A boolean. Used log scale on y-axis is set to True.

        Returns:
            None.

        """
        full_simulation_total_energy = self._full_simulation.calc_total_energy(
        )
        ml_simulation_total_energy = self._ml_simulation.calc_total_energy()

        plt.figure(figsize=(12, 8))
        bins = np.linspace(
            np.min(full_simulation_total_energy) -
            np.min(full_simulation_total_energy) * 0.05,
            np.max(full_simulation_total_energy) +
            np.max(full_simulation_total_energy) * 0.05, 50)
        plt.hist(x=full_simulation_total_energy,
                 histtype=HISTOGRAM_TYPE,
                 label="FullSim",
                 bins=bins,
                 color=FULL_SIM_HISTOGRAM_COLOR)
        plt.hist(x=ml_simulation_total_energy,
                 histtype=HISTOGRAM_TYPE,
                 label="MLSim",
                 bins=bins,
                 color=ML_SIM_HISTOGRAM_COLOR)
        plt.legend(loc="upper left")
        if y_log_scale:
            plt.yscale("log")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("# events")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._geometry}, {self._particle_theta} "
            + ("" if self._particle_phi is None else f", {self._particle_phi}")
        )
        plt.savefig(
            f"{self._save_dir}/E_tot_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png")
        )
        plt.clf()

    def _plot_cell_energy(self) -> None:
        """ Plots and saves a histogram with number of detector's cells across whole
        calorimeter with particular energy detected.

        Returns:
            None.

        """
        full_simulation_cell_energy = self._full_simulation.calc_cell_energy()
        ml_simulation_cell_energy = self._ml_simulation.calc_cell_energy()

        log_full_simulation_cell_energy = np.log10(
            full_simulation_cell_energy,
            out=np.zeros_like(full_simulation_cell_energy),
            where=(full_simulation_cell_energy != 0))
        log_ml_simulation_cell_energy = np.log10(
            ml_simulation_cell_energy,
            out=np.zeros_like(ml_simulation_cell_energy),
            where=(ml_simulation_cell_energy != 0))

        # Don't plot zeros (ml_simulation can have a threshold as well)
        log_full_simulation_cell_energy = log_full_simulation_cell_energy[log_full_simulation_cell_energy!=0]
        log_ml_simulation_cell_energy = log_ml_simulation_cell_energy[log_ml_simulation_cell_energy!=0]

        plt.figure(figsize=(12, 8))
        bins = np.linspace(-3, 4, 1000)
        plt.hist(x=log_full_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="FullSim",
                 color=FULL_SIM_HISTOGRAM_COLOR)
        plt.hist(x=log_ml_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="MLSim",
                 color=ML_SIM_HISTOGRAM_COLOR)
        plt.xlabel("log10(E/MeV)")
        plt.ylim(bottom=1)
        plt.yscale("log")
        plt.ylim(bottom=1)
        plt.ylabel("# entries")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._geometry}, {self._particle_theta} "
            + ("" if self._particle_phi is None else f", {self._particle_phi}")
        )
        plt.grid(True)
        plt.legend(loc="upper left")
        plt.savefig(
            f"{self._save_dir}/E_cell_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png")
        )
        plt.clf()

    def _plot_cell_energy_non_log(self) -> None:
        """ Plots and saves a histogram with number of detector's cells across whole
        calorimeter with particular energy detected.

        Returns:
            None.

        """
        full_simulation_cell_energy = self._full_simulation.calc_cell_energy()
        ml_simulation_cell_energy = self._ml_simulation.calc_cell_energy()

        plt.figure(figsize=(12, 8))
        #bins = np.linspace(1e-2, 4000, 1000)
        bins = np.linspace(1e-2, 400, 1000)
        plt.hist(x=full_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="FullSim",
                 color=FULL_SIM_HISTOGRAM_COLOR)
        plt.hist(x=ml_simulation_cell_energy,
                 bins=bins,
                 histtype=HISTOGRAM_TYPE,
                 label="MLSim",
                 color=ML_SIM_HISTOGRAM_COLOR)
        plt.xlabel("Energy [MeV]")
        plt.ylim(bottom=1)
        plt.yscale("log")
        plt.ylim(bottom=1)
        plt.ylabel("# entries")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._geometry}, {self._particle_theta} "
            + ("" if self._particle_phi is None else f", {self._particle_phi}")
        )
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.savefig(
            f"{self._save_dir}/E_cell_non_log_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png"))
        plt.xscale("log")
        plt.savefig(
            f"{self._save_dir}/E_cell_non_log_xlog_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png"))
        plt.clf()

    def _plot_zero_voxels(self) -> None:
        """ Plots and saves a histogram with number of detector's cells across whole
        calorimeter with particular energy detected.

        Returns:
            None.

        """
        full_simulation_cell_energy = self._full_simulation._input
        ml_simulation_cell_energy = self._ml_simulation._input

        full_simulation_cell_energy = full_simulation_cell_energy==0
        ml_simulation_cell_energy = ml_simulation_cell_energy==0

        full_sim_num_zeros_per_r = np.sum(full_simulation_cell_energy, axis=(0,2,3))
        ml_sim_num_zeros_per_r = np.sum(ml_simulation_cell_energy, axis=(0,2,3))

        plt.figure(figsize=(12, 8))
        plt.plot(range(DataInfo().N_CELLS_R), full_sim_num_zeros_per_r, color=FULL_SIM_HISTOGRAM_COLOR, label="FullSim")
        plt.plot(range(DataInfo().N_CELLS_R), ml_sim_num_zeros_per_r, color=ML_SIM_HISTOGRAM_COLOR, label="MLSim")
        plt.xlabel("")
        # plt.yscale("log")
        plt.ylabel("# Zeroes")
        plt.title(
            f" $e^-$, {self._particle_energy} [GeV], {self._geometry}, {self._particle_theta} "
            + ("" if self._particle_phi is None else f", {self._particle_phi}")
        )
        plt.grid(True)
        plt.legend(loc="upper right")
        plt.savefig(
            f"{self._save_dir}/Radial_num_zeroes_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png"))
        plt.clf()

    def _plot_energy_per_layer(self):
        """ Plots and saves N_CELLS_Z histograms with total energy detected in particular layers.

        Returns:
            None.

        """
        full_simulation_energy_per_layer = self._full_simulation.calc_energy_per_layer(
        )
        ml_simulation_energy_per_layer = self._ml_simulation.calc_energy_per_layer(
        )

        number_of_plots_in_row = 9
        number_of_plots_in_column = 5

        bins = np.linspace(np.min(full_simulation_energy_per_layer - 10),
                           np.max(full_simulation_energy_per_layer + 10), 25)

        fig, ax = plt.subplots(number_of_plots_in_column,
                               number_of_plots_in_row,
                               figsize=(20, 15),
                               sharex="all",
                               sharey="all",
                               constrained_layout=True)

        for layer_nb in range(DataInfo().N_CELLS_Z):
            i = layer_nb // number_of_plots_in_row
            j = layer_nb % number_of_plots_in_row

            ax[i][j].hist(full_simulation_energy_per_layer[:, layer_nb],
                          histtype=HISTOGRAM_TYPE,
                          label="FullSim",
                          bins=bins,
                          color=FULL_SIM_HISTOGRAM_COLOR)
            ax[i][j].hist(ml_simulation_energy_per_layer[:, layer_nb],
                          histtype=HISTOGRAM_TYPE,
                          label="MLSim",
                          bins=bins,
                          color=ML_SIM_HISTOGRAM_COLOR)
            ax[i][j].set_title(f"Layer {layer_nb}", fontsize=13)
            ax[i][j].set_yscale("log")
            ax[i][j].tick_params(axis='both', which='major', labelsize=10)

        fig.supxlabel("Energy [MeV]", fontsize=14)
        fig.supylabel("# entries", fontsize=14)
        fig.suptitle(
            f" $e^-$, {self._particle_energy} [GeV], {self._geometry}, {self._particle_theta} "
            + ("" if self._particle_phi is None else f", {self._particle_phi}")
        )

        # Take legend from one plot and make it a global legend.
        handles, labels = ax[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.5))

        plt.savefig(
            f"{self._save_dir}/E_layer_Geo_{self._geometry}_E_{self._particle_energy}_"
            + (f"Angle_{self._particle_theta}.png" if self._particle_phi is None
                else f"Theta_{self._particle_theta}_Phi_{self._particle_phi}.png"), bbox_inches="tight")
        plt.clf()

    def plot_and_save(self):
        """ Main plotting function.

        Calls private methods and prints the information about progress.

        Returns:
            None.

        """
        print("Plotting total energy...")
        self._plot_total_energy()
        print("Plotting cell energy...")
        self._plot_cell_energy()
        self._plot_cell_energy_non_log()
        self._plot_zero_voxels()
        print("Plotting energy per layer...")
        self._plot_energy_per_layer()
