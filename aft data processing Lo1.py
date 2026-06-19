# This is a sample Python script.

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.optimize import curve_fit

num_of_p_per_FFT_segment = 1024

cur_path = os.path.dirname(__file__)
multiple_path = os.path.relpath('data_for_PSD_plotting\\Multiple_Traces\\', cur_path)


def multiple_traces(specified_traces=None, use_original=True, use_potentials=False, ppsegment=1024, peak_info=False,
                    save_cor_traces=True, dump_traces=None, frequency_range_dplot=20000, frequancy_range_EvsS_area=2500):
    num_of_p_per_FFT_segment = ppsegment

    def double_exp(x, A1, tau1, A2, tau2):
        return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

    def robust_endpoint(values):
        """Average the 2 closest values out of the first 3."""
        v = values[:3]
        diffs = [abs(v[0]-v[1]), abs(v[0]-v[2]), abs(v[1]-v[2])]
        min_idx = np.argmin(diffs)
        pairs = [(v[0],v[1]), (v[0],v[2]), (v[1],v[2])]
        return np.mean(pairs[min_idx])

    for file_name in os.listdir(multiple_path):

        file_path = os.path.join(multiple_path, file_name)

        if not file_name.endswith(".atf"):
            continue

        print(file_name)

        with open(file_path) as f:
            lines = f.readlines()

        peak_noise_at_frequency = []
        frequency_of_peak_noise = []
        peak_value = 0
        peak_index = 0
        potential_step = 5
        potentials_used = []
        mean_noise_above_peak = []

        header_line = lines[12].split()
        num_traces = (len(header_line) - 1) // 2

        time_values = np.array([float(lines[x].split()[0]) for x in range(11, len(lines))])
        potential_step = np.round([float(lines[11].split()[4]) - float(lines[11].split()[2])])[0]
        print(potential_step, 'mV steps')
        dt = time_values[1] - time_values[0]
        print('dt = ', dt)
        sampling_frequency = round(1.0 / dt)
        print('sampling_frequency = ', sampling_frequency)

        frequency_step = sampling_frequency / num_of_p_per_FFT_segment
        print('frequency_step = ',frequency_step)
        dplot_range = round(frequency_range_dplot / frequency_step)
        area_range = round(frequancy_range_EvsS_area / frequency_step)
        
        frequency_index_1 = 25
        frequency_index_2 = 300

        num_points = len(lines) - 11
        currents = np.zeros((num_traces, num_points))
        potentials = np.zeros((num_traces, num_points))

        for row in range(11, len(lines)):
            cols = lines[row].split()
            for t in range(1, num_traces):
                currents[t - 1, row - 11] = float(cols[t * 2 - 1])
                potentials[t - 1, row - 11] = float(cols[t * 2])

        if dump_traces is not None:
            for i in dump_traces:
                with open(f'{file_path} trace {i} .csv', 'w', ) as myfile:
                    field_labels = ['time, s', 'current, pA', 'potential, mV']
                    wr = csv.writer(myfile, lineterminator='\n')
                    wr.writerow([field_labels[0], field_labels[1], field_labels[2]])
                    print(time_values)
                    for entity in range(num_points):
                        etime = float(time_values[entity])
                        print(etime)
                        ecurrent = currents[i, entity]
                        epotential = potentials[i, entity]
                        wr.writerow([etime, ecurrent, epotential])

        corrected_traces = np.zeros_like(currents)

        noise_values = []
        min_cutoff = 1e-10
        max_Pxx = sec_Pxx = third_Pxx = 0
        max_trace = sec_trace = third_trace = 0
        num_cutoff = 0

        trace_list = []
        for i in range(num_traces):
            trace_list.append(i)

        if specified_traces is not None:
            trace_list = specified_traces
            num_traces = len(trace_list)

        corrected_traces = np.zeros_like(currents)
        exponent_fits = np.zeros_like(currents)
        tau_params = np.zeros_like(trace_list)
        exp_covariance = np.zeros_like(trace_list)
        zero_traces = []

        if use_potentials:
            max_Pxx = sec_Pxx = third_Pxx = 0
            max_trace = sec_trace = third_trace = 0
            for i in trace_list:
                trace = potentials[i, :]
                f, Pxx = signal.welch(trace, fs=sampling_frequency, detrend='linear', nperseg=num_of_p_per_FFT_segment)
                plt.loglog(f, Pxx, label="trace #" + str(i), linewidth=0.5)
                plt.title('V noise')
                if num_traces <= 7:
                    plt.legend()
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('PSD Potential [mV^2/Hz')
            psd_path = os.path.join(multiple_path, file_name.replace(".atf", "Voltage.png"))
            plt.savefig(psd_path, dpi=800, bbox_inches="tight")
            plt.clf()

        if use_original:
            for i in trace_list:
                trace = currents[i, :]
                x = np.arange(num_points)

                if num_traces <= 7:
                    try:
                        popt, pcov = curve_fit(
                            double_exp, x, trace,
                            p0=[trace[0], 1, trace[0] / 2, 1],
                            maxfev=20000
                        )
                        background = double_exp(x, *popt)
                        corrected = trace - background
                        tau_params[i] = popt
                        exp_covariance[i] = pcov
                        exponent_fits[i, :] = background
                    except Exception:
                        corrected = trace
                        print("exception in the fitting! Corrected = Original Trace")
                    if corrected[1] == 0:
                        print(f'trace #{i} is zero')
                        zero_traces.append(i)
                    corrected_traces[i, :] = corrected

                f, Pxx = signal.welch(trace, fs=sampling_frequency, detrend='linear', nperseg=num_of_p_per_FFT_segment)

                freq_used = "{:.3f}".format(f[1])
                mean_noise_above_peak.append(np.mean(Pxx[frequency_index_2:]))
                peak_value = Pxx[frequency_index_1:frequency_index_2].max()
                peak_index = np.where(Pxx == peak_value)[0][0]
                peak_noise_at_frequency.append(Pxx[peak_index])
                potential_of_this_peak = i * potential_step
                potentials_used.append(potential_of_this_peak)
                print('trace #', i, ', potential = ', potential_of_this_peak)
                print('peak noise value equals = ', peak_value)
                print('at index = ', peak_index)
                frequency_of_peak_noise.append(peak_index * frequency_step)

                if Pxx[1] > max_Pxx:
                    third_Pxx, third_trace = sec_Pxx, sec_trace
                    sec_Pxx, sec_trace = max_Pxx, max_trace
                    max_Pxx, max_trace = Pxx[1], i

                if Pxx[1] > min_cutoff:
                    plt.loglog(f, Pxx, label="trace #" + str(i), linewidth=2)
                    noise_values.append(Pxx[1])
                else:
                    num_cutoff += 1

            plt.title(file_name)
            if num_cutoff > 0:
                plt.figtext(0.4, 0.15, f"Traces below threshold: {num_cutoff}")
            plt.xlabel("frequency [Hz]")
            plt.ylabel("PSD [pA²/Hz]")
            max_Pxx = "{:.3f}".format(max_Pxx)
            sec_Pxx = "{:.3f}".format(sec_Pxx)
            third_Pxx = "{:.3f}".format(third_Pxx)
            plt.figtext(0.4, 0.2,
                        f"max = {max_Pxx} at trace {max_trace}\n"
                        f"second = {sec_Pxx} at trace {sec_trace}\n"
                        f"third = {third_Pxx} at trace {third_trace}")
            plt.figtext(0.4, 0.13, f"number of traces = {num_traces}")
            if num_traces <= 7:
                plt.legend()
            psd_path = os.path.join(multiple_path, file_name.replace(".atf", "Original.png"))
            plt.savefig(psd_path, dpi=800, bbox_inches="tight")
            plt.clf()

            plt.semilogy(range(len(noise_values)), noise_values, "r-")
            plt.xlabel("Trace No")
            plt.ylabel("Noise, pA²/Hz")
            plt.figtext(0.5, 0.7, f"At frequency = {freq_used}")
            plt.title('PSD from original traces')
            evs_path = os.path.join(multiple_path, file_name.replace(".atf", " Original E vs S.png"))
            plt.savefig(evs_path, dpi=800, bbox_inches="tight")
            plt.clf()

            time_axis = np.arange(num_points) * dt
            plt.figure(figsize=(12, 6))
            for i in trace_list:
                plt.plot(time_axis, currents[i, :], alpha=0.6, label="trace #" + str(i), linewidth=0.5)
                if num_traces <= 7:
                    plt.plot(time_axis, exponent_fits[i, :], label="double exp" + str(i), linewidth=0.5, color="red")
            plt.xlabel("Time (s)")
            plt.ylabel("Current (pA)")
            plt.title(f"All Original Traces — {file_name}")
            if num_traces <= 7:
                plt.legend()
            orig_all_path = os.path.join(multiple_path, f"{file_name}_ALL_original_traces.png")
            plt.savefig(orig_all_path, dpi=300, bbox_inches="tight")
            plt.close()

            if peak_info:
                potentials_used = np.ma.masked_equal(potentials_used, 0)
                plt.plot(potentials_used, frequency_of_peak_noise, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Frequency, Hz")
                plt.title(f" Original frequency of peak versus potentials")
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", " Original Peak Frequency Shift.png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()

                print(mean_noise_above_peak)
                plt.plot(potentials_used, mean_noise_above_peak, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Mean noise, pA^2/Hz")
                plt.title(f' Original Mean noise above index {frequency_index_2}')
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", "Original Mean Noise above peak.png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()

                plt.plot(potentials_used, peak_noise_at_frequency, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Noise of the Peak, pA^2/Hz")
                plt.title(f" Original Peak noise between indices {frequency_index_1} and {frequency_index_2} ")
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", "Original Noise amplitude shift .png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()

        # --- Reset analysis variables ---
        noise_values = []
        min_cutoff = 1e-10
        max_Pxx = sec_Pxx = third_Pxx = 0
        max_trace = sec_trace = third_trace = 0
        num_cutoff = 0
        zero_traces = []
        potentials_used = []
        peak_noise_at_frequency = []
        frequency_of_peak_noise = []
        mean_noise_above_peak = []

        freq_used = "{:.3f}".format(f[1])
        mean_noise_above_peak.append(np.mean(Pxx[frequency_index_2:]))
        peak_value = Pxx[frequency_index_1:frequency_index_2].max()
        peak_index = np.where(Pxx == peak_value)[0][0]
        peak_noise_at_frequency.append(Pxx[peak_index])
        potential_of_this_peak = i * potential_step
        potentials_used.append(potential_of_this_peak)
        print('trace #', i, ', potential = ', potential_of_this_peak)
        print('peak noise value equals = ', peak_value)
        print('at index = ', peak_index)
        frequency_of_peak_noise.append(peak_index * frequency_step)

        if Pxx[1] > max_Pxx:
            third_Pxx, third_trace = sec_Pxx, sec_trace
            sec_Pxx, sec_trace = max_Pxx, max_trace
            max_Pxx, max_trace = Pxx[1], i

        if Pxx[1] > min_cutoff:
            plt.loglog(f, Pxx, label="trace #" + str(i), linewidth=2)
            noise_values.append(Pxx[1])
        else:
            num_cutoff += 1
            plt.title(f'{file_name} Corrected')
            if num_cutoff > 0:
                plt.figtext(0.4, 0.15, f"Traces below threshold: {num_cutoff}")
            plt.xlabel("frequency [Hz]")
            plt.ylabel("PSD [pA²/Hz]")
            if num_traces <= 7:
                plt.legend()
            max_Pxx = "{:.3f}".format(max_Pxx)
            sec_Pxx = "{:.3f}".format(sec_Pxx)
            third_Pxx = "{:.3f}".format(third_Pxx)
            plt.figtext(0.4, 0.2,
                        f"max = {max_Pxx} at trace {max_trace}\n"
                        f"second = {sec_Pxx} at trace {sec_trace}\n"
                        f"third = {third_Pxx} at trace {third_trace}")
            plt.figtext(0.4, 0.13, f"number of traces = {num_traces}")
            psd_path = os.path.join(multiple_path, file_name.replace(".atf", "Corrected.png"))
            plt.savefig(psd_path, dpi=800, bbox_inches="tight")
            plt.clf()

            plt.semilogy(range(len(noise_values)), noise_values, "r-")
            plt.xlabel("Trace No")
            plt.ylabel("Noise, pA²/Hz")
            plt.title('PSD from corrected traces')
            plt.figtext(0.5, 0.7, f"At frequency = {freq_used}")
            evs_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected E vs S.png"))
            plt.savefig(evs_path, dpi=800, bbox_inches="tight")
            plt.clf()

            # --- Collect per-trace PSD using UNCORRECTED traces for area/E vs S plots ---
            all_Pxx = []
            for i in trace_list:
                f, Pxx = signal.welch(
                    currents[i, :],
                    fs=sampling_frequency,
                    detrend='linear',
                    nperseg=num_of_p_per_FFT_segment
                )
                all_Pxx.append(Pxx)

            all_Pxx = np.array(all_Pxx)
            summed_Pxx = np.sum(all_Pxx, axis=0)
            n_bins = len(summed_Pxx)
            baseline = np.linspace(summed_Pxx[0], summed_Pxx[-1], n_bins)
            corrected_area = summed_Pxx - baseline

            plt.figure()
            plt.plot(f, corrected_area, linewidth=1)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Baseline-subtracted summed PSD [pA²/Hz]")
            plt.title(f"{file_name} — Summed PSD Area (Baseline Subtracted)")
            area_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected Area vs Frequency.png"))
            plt.savefig(area_path, dpi=800, bbox_inches="tight")
            plt.close()

            plt.close('all')
            area_per_bin = []

            for k in range(1, area_range+1):
                noise_at_bin_k = np.array([all_Pxx[t, k] for t in range(len(trace_list))])
                left = robust_endpoint(noise_at_bin_k)
                right = robust_endpoint(noise_at_bin_k[::-1])
                baseline_k = np.exp(np.linspace(np.log(left), np.log(right), len(trace_list)))
                area = np.trapezoid(noise_at_bin_k - baseline_k, x=trace_list)
                area_per_bin.append(area)
                print("generating E vs S plot ", k)
                plt.figure()
                plt.semilogy(trace_list, noise_at_bin_k, "r-", label="PSD")
                plt.semilogy(trace_list, baseline_k, "b--", label="baseline")
                plt.xlabel("Trace No")
                plt.ylabel("Noise, pA²/Hz")
                plt.title(f"{file_name} — E vs S at {f[k]:.3f} Hz")
                plt.figtext(0.5, 0.7, f"At frequency = {f[k]:.3f} Hz")
                plt.legend()
                evs_bin_path = os.path.join(multiple_path, file_name.replace(".atf", f" Corrected E vs S bin {k} ({f[k]:.1f} Hz).png"))
                plt.savefig(evs_bin_path, dpi=300, bbox_inches="tight")
                plt.close()

            plt.figure()
            plt.plot(f[1:area_range+1], area_per_bin, linewidth=1)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Area above baseline [pA²/Hz · traces]")
            plt.title(f"{file_name} — Area above baseline vs Frequency")
            area_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected Area vs Frequency.png"))
            plt.savefig(area_path, dpi=800, bbox_inches="tight")
            plt.close()

            # --- 3D surface plot: frequency vs trace number vs log noise ---
            freq_vals = f[1: dplot_range+1]
            trace_vals = np.array(trace_list)
            Z = np.array([[all_Pxx[t, k] for t in range(len(trace_list))] for k in range(1, dplot_range+1)])
            Z_log = np.log10(Z)
            X, Y = np.meshgrid(trace_vals, freq_vals)
            
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_log, cmap='viridis', linewidth=0, antialiased=True)
            ax.set_xlabel("Trace No")
            ax.set_ylabel("Frequency [Hz]")
            ax.set_zlabel("log10(PSD) [pA²/Hz]")
            ax.set_title(f"{file_name} — 3D Noise Surface")
            ax.invert_yaxis()
            fig.colorbar(surf, ax=ax, shrink=0.5, label="log10(PSD)")
            surface_path = os.path.join(multiple_path, file_name.replace(".atf", " 3D Noise Surface.png"))
            plt.savefig(surface_path, dpi=300, bbox_inches="tight")
            plt.close()

            time_axis = np.arange(num_points) * dt
            plt.figure(figsize=(12, 6))
            for i in trace_list:
                plt.plot(time_axis, corrected_traces[i, :], alpha=0.6, label="trace #" + str(i), linewidth=0.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Current (pA)")
            plt.title(f"All Corrected Traces — {file_name}")
            if num_traces <= 7:
                plt.legend()
            corr_all_path = os.path.join(multiple_path, f"{file_name}_ALL_corrected_traces.png")
            plt.savefig(corr_all_path, dpi=300, bbox_inches="tight")
            plt.close()

            if peak_info:
                potentials_used = np.ma.masked_equal(potentials_used, 0)

                plt.plot(potentials_used, frequency_of_peak_noise, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Frequency, Hz")
                plt.title(f" Corrected frequency of peak versus potentials")
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected Peak Frequency Shift.png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()

                print('mean noise above peak: ', mean_noise_above_peak)
                print('potentials used = ', potentials_used)
                plt.plot(potentials_used, mean_noise_above_peak, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Mean noise, pA^2/Hz")
                plt.title(f'Corrected Mean noise above index {frequency_index_2}')
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected Mean Noise above peak.png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()

                plt.plot(potentials_used, peak_noise_at_frequency, "c-")
                plt.xlabel("Potential (mV)")
                plt.ylabel("Noise of the Peak, pA^2/Hz")
                plt.title(f"Corrected Peak noise between indices {frequency_index_1} and {frequency_index_2} ")
                peak_at_f_path = os.path.join(multiple_path, file_name.replace(".atf", " Corrected Noise amplitude shift .png"))
                plt.savefig(peak_at_f_path, dpi=800, bbox_inches="tight")
                plt.close()


multiple_traces(specified_traces=None, use_original=True, use_potentials=True, ppsegment=1024*4, peak_info=True,
                save_cor_traces=False, dump_traces=None,frequency_range_dplot=2500, frequancy_range_EvsS_area=2500)