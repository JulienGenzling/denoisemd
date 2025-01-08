import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import deque

class EMD:
    """
    Empirical Mode Decomposition (EMD) class implementing the EMD algorithm along with various denoising and detrending methods.
    """
    def __init__(self, 
                 max_imf=10, 
                 max_sifting_iter=10, 
                 std_thr=0.2, 
                 total_power_thr=0.005, 
                 theta1=0.05,
                 theta2=0.5,
                 alpha=0.05,
                 ):
        """
        Initialize EMD parameters.

        Parameters:
        - max_imf: Maximum number of IMFs to extract.
        - max_sifting_iter: Maximum iterations allowed for sifting.
        - std_thr: Standard deviation threshold for stopping criteria.
        - total_power_thr: Residual power threshold for decomposition.
        - theta1, theta2, alpha: Parameters for Rilling stopping criteria.
        """
        self.max_imf = max_imf
        self.max_sifting_iter = max_sifting_iter
        self.std_thr = std_thr
        self.total_power_thr = total_power_thr
        
        # For Rilling test http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf
        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha = alpha
    
        # Interpolation from the table in https://perso.ens-lyon.fr/patrick.flandrin/HHT05.pdf
        self.H_table = np.array([0.2, 0.5, 0.8])
        self.beta_table = np.array([0.487, 0.719, 1.025])
        self.a_95_table = np.array([0.458, 0.474, 0.497])
        self.b_95_table = np.array([-2.435, -2.449, -2.331])
        self.a_99_table = np.array([0.452, 0.460, 0.495])
        self.b_99_table = np.array([-1.951, -1.919, -1.833])

        self.beta_interp  = interp1d(self.H_table, self.beta_table, kind='linear', fill_value='extrapolate')
        self.a_95_interp  = interp1d(self.H_table, self.a_95_table,  kind='linear', fill_value='extrapolate')
        self.b_95_interp  = interp1d(self.H_table, self.b_95_table,  kind='linear', fill_value='extrapolate')
        self.a_99_interp  = interp1d(self.H_table, self.a_99_table,  kind='linear', fill_value='extrapolate')
        self.b_99_interp  = interp1d(self.H_table, self.b_99_table,  kind='linear', fill_value='extrapolate')

    def _get_extrema(self, signal):
        """
        Identify local maxima, minima, and zero-crossings in `signal`.
        """
        diff = np.diff(signal)
        increasing = diff > 0
        decreasing = diff < 0
        max_indices = np.where(np.logical_and(increasing[:-1], decreasing[1:]))[0] + 1
        min_indices = np.where(np.logical_and(decreasing[:-1], increasing[1:]))[0] + 1

        zero_indices = np.where(np.diff(np.sign(signal)) != 0)[0]
        return max_indices, min_indices, zero_indices

    def mirror_left(self, signal, t_max, t_min, nb_mirror):  
        """
        Perform left-side signal mirroring for spline padding.
        """      
        if t_max[0] < t_min[0]:
            if signal[0] > signal[t_min[0]]:
                x1 = t_max[1:nb_mirror+1][::-1]
                y1 = t_min[:nb_mirror][::-1]
                p = t_max[0]
            else:
                x1 = t_max[:nb_mirror][::-1]
                y1 = np.append(t_min[:nb_mirror-1][::-1], 0)
                p = 0
        else:
            if signal[0] < signal[t_max[0]]:
                x1 = t_max[:nb_mirror][::-1]
                y1 = t_min[1:nb_mirror+1][::-1]
                p = t_min[0]
            else:
                x1 = np.append(t_max[:nb_mirror-1][::-1], 0)
                y1 = t_min[:nb_mirror][::-1]
                p = 0
        return 2*p - x1, signal[x1], 2*p - y1, signal[y1]
    
    def mirror_right(self, signal, t_max, t_min, nb_mirror):
        """
        Perform right-side signal mirroring for spline padding.
        """
        emax, emin = len(t_max), len(t_min)
        if t_max[-1] < t_min[-1]:
            if signal[-1] < signal[t_max[-1]]:
                x2 = t_max[emax - nb_mirror:][::-1]
                y2 = t_min[emin - nb_mirror - 1:-1][::-1]
                p = t_min[-1]
            else:
                x2 = np.append(t_max[emax - nb_mirror + 1:], len(signal) - 1)[::-1]
                y2 = t_min[emin - nb_mirror:][::-1]
                p = len(signal) - 1
        else:
            if signal[-1] > signal[t_min[-1]]:
                x2 = t_max[emax - nb_mirror - 1:-1][::-1]
                y2 = t_min[emin - nb_mirror:][::-1]
                p = t_max[-1]
            else:
                x2 = t_max[emax - nb_mirror:][::-1]
                y2 = np.append(t_min[emin - nb_mirror + 1:], len(signal) - 1)[::-1]
                p = len(signal) - 1
        return 2*p - x2, signal[x2], 2*p - y2, signal[y2]

    def _get_envelope(self, signal, nb_mirror = 2):
        """
        Compute upper and lower envelopes using cubic splines.
        """
        n = len(signal)
        t = np.arange(n)
        max_indices, min_indices, _ = self._get_extrema(signal)
                
        t_max, s_max = t[max_indices], signal[max_indices]
        t_min, s_min = t[min_indices], signal[min_indices]

        # We pad the extremums to improve the spline interpolation
        # We use the same logic as the matlab code provided in https://perso.ens-lyon.fr/patrick.flandrin/emd.html.
        t_max_left, s_max_left, t_min_left, s_min_left = self.mirror_left(signal, t_max, t_min, nb_mirror)
        t_max_right, s_max_right, t_min_right, s_min_right = self.mirror_right(signal, t_max, t_min, nb_mirror)
 
        t_max_all = np.concatenate([t_max_left, t_max, t_max_right])
        s_max_all = np.concatenate([s_max_left, s_max, s_max_right])

        t_min_all = np.concatenate([t_min_left, t_min, t_min_right])
        s_min_all = np.concatenate([s_min_left, s_min, s_min_right])

        i1 = np.argsort(t_max_all)
        t_max_all, s_max_all = t_max_all[i1], s_max_all[i1]
        i2 = np.argsort(t_min_all)
        t_min_all, s_min_all = t_min_all[i2], s_min_all[i2]

        upper_env_spline, lower_env_spline = CubicSpline(t_max_all, s_max_all), CubicSpline(t_min_all, s_min_all)
        lower_env_spline = CubicSpline(t_min_all, s_min_all)

        upper_env, lower_env = upper_env_spline(t), lower_env_spline(t)

        return upper_env, lower_env  

    
    def _is_end_imf(self, verbose, **kwargs):
        """
        Check whether the sifting process should stop.
        """
        # ext and zeros test
        if "nb_ext" in kwargs and "nb_zeros" in kwargs:
            if abs(kwargs["nb_ext"] - kwargs["nb_zeros"]) <= 1:
                if verbose:
                    print("Stopped - nb zeros vs ext")
                return True
        
        # Rilling test http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf
        if "upper_env" in kwargs and "lower_env" in kwargs:
            mean_env = (kwargs["upper_env"]+kwargs["lower_env"])/2
            a = (kwargs["upper_env"]-kwargs["lower_env"])/2
            sigma = np.abs(mean_env/a)
            fraction_below_theta1 = np.mean(sigma < self.theta1)
            if fraction_below_theta1 >= (1 - self.alpha) and np.all(sigma < self.theta2):
                if verbose:
                    print("Stopped - Rilling")
                return True
        
        # Standard deviation test https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.1998.0193
        if "imf_old" in kwargs and "imf" in kwargs:
            imf_diff = kwargs["imf"] - kwargs["imf_old"]
            std = np.sum((imf_diff / kwargs["imf"]) ** 2)
            if std < self.std_thr:
                if verbose:
                    print("Stopped - SD")
                return True
            
        
        # max sifting iter test
        if "n_iter" in kwargs:
            if kwargs["n_iter"] == self.max_sifting_iter:
                if verbose:
                    print("Stopped - max iter")
                return True
        
        return False
    
    def _is_end(self, residual, reached_trend, verbose):
        """
        Check whether the decomposition process should.
        """
        l1 = np.sum(np.abs(residual)) < self.total_power_thr
        if verbose:
            print(l1, reached_trend)
        return l1 or reached_trend

    def _sift(self, signal, verbose):
        """
        Perform the sifting process to extract one IMF.
        """
        imf = signal.copy()
        reached_trend = False
        n_iter = 0
        while True:
            max_pos, min_pos, _ = self._get_extrema(imf)
            if len(max_pos) + len(min_pos) <= 2:
                reached_trend = True
                break
            else:
                upper_env, lower_env = self._get_envelope(imf)
                mean_env = (upper_env + lower_env) / 2
                imf_old = imf.copy()
                imf = imf_old - mean_env
                max_pos, min_pos, zeros = self._get_extrema(imf)
                if self._is_end_imf(
                                    verbose,
                                    upper_env=upper_env, 
                                    lower_env=lower_env, 
                                    imf_old=imf_old, 
                                    imf=imf, 
                                    # nb_ext=len(max_pos)+len(min_pos), 
                                    # nb_zeros=len(zeros),
                                    n_iter=n_iter,
                                    ):
                    break
            n_iter += 1
        return imf, reached_trend

    def decompose(self, signal, verbose=0):
        """
        Decompose a signal into IMFs using EMD.

        Returns:
        - imfs: Extracted IMFs.
        - residual: Residual signal after decomposition.

        """
        imfs = np.empty((0, len(signal)))
        residual = signal.copy()
        stop = False
        while len(imfs) < self.max_imf and not stop:
            imf, reached_trend = self._sift(residual, verbose)
            imfs = np.vstack((imfs, imf))
            residual -= imf
            stop = self._is_end(residual, reached_trend, verbose)
        return imfs, residual # IMFS, residual

    @staticmethod
    def plot_decomp(signal, imfs, residual):
        """
        Plot the original signal, extracted IMFs, and residual.
        """
        imfs = np.vstack((imfs, residual))
        imfNo = len(imfs)

        fig_height = 1.5 * (imfNo + 1)  
        plt.figure(figsize=(8, fig_height))

        plt.subplot(imfNo + 1, 1, 1)
        plt.plot(np.arange(len(signal)), signal, "r")
        plt.title("Original Signal")

        for num in range(imfNo):
            plt.subplot(imfNo + 1, 1, num + 2)
            plt.plot(np.arange(len(signal)), imfs[num], "g")
            plt.ylabel(f"IMF {num + 1}")

        plt.subplot(imfNo + 1, 1, imfNo + 1)
        plt.plot(np.arange(len(signal)), residual, "b")
        plt.ylabel("Residual")

        plt.show()

    def _get_standardized_means(self, imfs):
        partial = np.zeros_like(imfs[0])
        means = []
        for D in range(len(imfs)):
            partial += imfs[D]
            mu = np.mean(partial)
            sigma = np.std(partial) + 1e-12
            standardized_mean = mu / sigma
            means.append(standardized_mean)
        return means
    
    def _plot_standardized_means(self, means, threshold):
        plt.plot(np.arange(1, len(means)+1), means, 'o-')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.axhline(y=threshold, color='red', linestyle='--')
        plt.axhline(y=-threshold, color='red', linestyle='--')
        plt.xlabel('d')
        plt.ylabel('Standardized means')
        plt.title('fine-to-coarse EMD reconstruction')
        plt.show()

    def detrend(self, T, signal, imfs, residual, threshold, plot=True):
        """
        Detrend a signal by separating the trend from IMFs.
        """
        means = self._get_standardized_means(imfs)
        self._plot_standardized_means(means, threshold)
        for D in range(len(means)):
            mean = means[D]
            if abs(mean) >= threshold:
                break
            elif D == len(means) - 1:
                D += 1

        plt.plot(T, signal, label='Original Signal')
        plt.plot(T, residual + sum(imfs[D:]), '-.', label=f'trend')
        plt.plot(T, sum(imfs[:D]), '--', label=f'detrended')
        plt.title(f'Signal detrending with D={D} as the change point')
        plt.legend()
        plt.show()

    def _get_conf_params(self, H, conf=0.95):
        betaH = float(self.beta_interp(H))

        if conf == 0.95:
            aH = float(self.a_95_interp(H))
            bH = float(self.b_95_interp(H))
        elif conf == 0.99:
            aH = float(self.a_99_interp(H))
            bH = float(self.b_99_interp(H))
        else:
            raise ValueError(f"conf={conf} not recognized")

        return betaH, aH, bH

    def denoise(self, T, signal, imfs, residual, ground_truth=None, H=0.3, conf=0.99, plot=True):
        """
        Perform denoising using a noise energy model.
        
        Returns:
        - partial: Denoised signal.
        - noise_estimate: Estimated noise.
        """
        betaH, aH, bH = self._get_conf_params(H, conf)

        # eq. (6)
        E1 = np.sum(imfs[0]**2) 
        W_H1 = E1

        # eq. (7)
        rho_H = 2.0
        CH = W_H1 / betaH
        def W_H(k):
            if k == 1:
                return W_H1
            return CH * (rho_H ** (-2 * (1 - H) * k))

        # eq. (5)
        def T_H(k):
            inner = aH * k + bH
            return W_H(k) * 2.0 ** (2.0 ** inner)

        energies = [np.sum(imf**2) for imf in imfs]
        keep_flags = []
        for i, E_i in enumerate(energies, start=1):
            if E_i > T_H(i):
                keep_flags.append(True)
            else:
                keep_flags.append(False)

        partial = residual.copy()
        for (flag, imf) in zip(keep_flags, imfs):
            if flag:
                partial += imf

        noise_estimate = np.zeros_like(residual)
        for (flag, imf) in zip(keep_flags, imfs):
            if not flag:
                noise_estimate += imf

        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            k_vals = np.arange(1, len(imfs)+1)
            log2_energies = np.log2(energies)
            model_curve = [np.log2(W_H(k)) for k in k_vals]
            conf_curve  = [np.log2(T_H(k)) for k in k_vals]

            axes[0,0].plot(k_vals, log2_energies, 'o-', label='Signal IMFs')
            axes[0,0].plot(k_vals, model_curve, 'x--', label=f'Model (H={H:.2f})')
            axes[0,0].plot(k_vals, conf_curve, 'r--', label=f'{int(conf*100)}% CI')
            axes[0,0].set_xlabel('IMF index')
            axes[0,0].set_ylabel('log2(energy)')
            axes[0,0].legend()
            axes[0,0].set_title('IMF energies vs. noise model')

            axes[0,1].plot(T, signal, 'k')
            axes[0,1].set_title('Original signal')

            axes[1,0].plot(T, partial, 'b', label='Denoised')
            if ground_truth is not None:
                axes[1,0].plot(T, ground_truth, 'k--', label='Ground truth')
            axes[1,0].legend()
            axes[1,0].set_title('Denoised signal')

            axes[1,1].plot(T, noise_estimate, 'g')
            axes[1,1].set_title('Noise estimate')

            plt.tight_layout()
            plt.show()

        return partial, noise_estimate
    
    def _morphological_reconstruct_1d(self, marker, mask):
        """
        Perform morphological reconstruction in 1D using geodesic dilation.
        """
        out = marker.copy()
        queue = deque(range(len(out)))

        while queue:
            idx = queue.popleft()
            for neighbor in (idx - 1, idx + 1):
                if 0 <= neighbor < len(out):
                    new_val = min(mask[neighbor], out[idx])
                    if new_val > out[neighbor]:
                        out[neighbor] = new_val
                        queue.append(neighbor)
        return out

    def degressive_denoise(self, T, signal, imfs, residual, start_var=2., ground_truth=None, plot=True):
        """
        Degressive denoising using morphological reconstruction based on https://dev.ipol.im/~morel/M%E9moires_Stage_Licence_2011/Maud%20Kerebel%2C%20Luc%20Pellissier%2C%20Daniel%20StanMemoire_EMD.pdf.

        Returns:
        - denoised: Denoised signal.
        - thresholds: Thresholds applied to each IMF.
        """
        n_imfs = len(imfs)
        std_list = [np.std(imf) for imf in imfs]
        if n_imfs > 1:
            alpha_vals = np.linspace(start_var, 0.0, n_imfs)
        else:
            alpha_vals = np.array([start_var]) 

        thresholds = []
        reconstructed_imfs = []

        for i, imf in enumerate(imfs):
            alpha_i = alpha_vals[i]
            std_i = std_list[i]
            thr_i = alpha_i * std_i
            thresholds.append(thr_i)

            mask = imf
            marker = np.where(np.abs(imf) >= thr_i, imf, 0.0)

            recon_imf = self._morphological_reconstruct_1d(marker, mask)
            reconstructed_imfs.append(recon_imf)

        denoised = np.sum(reconstructed_imfs, axis=0)

        if plot:
            fig, axs = plt.subplots(n_imfs + 2, 1, figsize=(8, 2 * (n_imfs + 2)), sharex=True)
            axs[0].plot(T, signal, 'k', label='Original Signal')
            axs[0].set_title("Original Signal")
            axs[0].legend()

            for i, recon_imf in enumerate(reconstructed_imfs):
                axs[i + 1].plot(T, recon_imf, label=f'Reconstructed IMF {i + 1}')
                axs[i + 1].axhline(y=thresholds[i], color='r', linestyle='--', label=f'Threshold: {thresholds[i]:.2f}')
                axs[i + 1].axhline(y=-thresholds[i], color='r', linestyle='--', label=f'Threshold: {thresholds[i]:.2f}')
                axs[i + 1].set_title(f"Reconstructed IMF {i + 1}")
                axs[i + 1].legend()

            axs[-1].plot(T, denoised, 'b', label='Degressive Denoised Signal')
            if ground_truth is not None:
                axs[-1].plot(T, ground_truth, 'k--', label='Ground Truth')
            axs[-1].set_title("Denoised signal")
            axs[-1].legend()

            plt.tight_layout()
            plt.show()
        
        return denoised, thresholds
    
    def thresholdinf_denoise(self, signal, imfs, residual):
        """
        Perform threshold-based denoising of IMFs based on https://www.researchgate.net/publication/224317143_Speech_Signal_Noise_Reduction_by_EMD.

        Returns:
        - denoised_signal: Reconstructed signal after denoising.
        """
        C = imfs.shape[0]
        T = len(signal)
        
        sigma1 = 1.4826 * np.median(np.abs(imfs[1] - np.median(imfs[1])))
        
        denoised_imfs = np.zeros_like(imfs)
        
        for j in range(C):
            sigma_j = sigma1 / (np.sqrt(2) ** j)
            tau_j = np.sqrt(2 * np.log(T)) * sigma_j
            denoised_imfs[j] = np.where(
                np.abs(imfs[j]) > tau_j,
                imfs[j],
                0
            )
            # denoised_imfs[j] = np.sign(imfs[j]) * np.maximum(0, np.abs(imfs[j]) - tau_j)
        
        denoised_signal = np.sum(denoised_imfs, axis=0) + residual
        
        return denoised_signal
        