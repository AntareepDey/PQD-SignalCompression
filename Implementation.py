import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft, ifft
from sklearn.linear_model import OrthogonalMatchingPursuit
from collections import Counter

class PQDSignalCompressor:
    def __init__(self, fs=6400, duration=0.1, snr_levels=[30, 40, 50]):
        self.fs = fs
        self.duration = duration
        self.snr_levels = snr_levels

    def add_noise(self, signal, snr):
        power_signal = np.mean(signal ** 2)
        power_noise = power_signal / (10 ** (snr / 10))
        noise = np.sqrt(power_noise) * np.random.randn(len(signal))
        return signal + noise

    def generate_signal(self, signal_type, snr):
        t = np.arange(0, self.duration, 1 / self.fs)
        if signal_type == "impulse":
            signal = np.sin(2 * np.pi * 50 * t)
            impulse_center = int(0.06 * self.fs)
            impulse_width = int(0.001 * self.fs)
            signal[impulse_center:impulse_center + impulse_width] += np.exp(-np.arange(impulse_width))
        elif signal_type == "sag":
            signal = np.sin(2 * np.pi * 50 * t)
            sag_start, sag_end = int(0.04 * self.fs), int(0.08 * self.fs)
            signal[sag_start:sag_end] *= 0.5
        elif signal_type == "decaying_harmonics":
            signal = np.exp(-2 * np.pi * t) * (
                np.sin(2 * np.pi * 50 * t) +
                0.5 * np.sin(2 * np.pi * 150 * t + np.pi / 4) +
                0.3 * np.sin(2 * np.pi * 250 * t + np.pi / 3)
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        return self.add_noise(signal, snr)

    def _hartley_transform_matrix(self, n):
        """Generate the Hartley transform matrix."""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = np.cos(2 * np.pi * i * j / n) + np.sin(2 * np.pi * i * j / n)
        return H / np.sqrt(n)

    def sparse_decomposition(self, signal):
        """Separate signal into Transient Component (TC) and Steady-State Component (SSC)."""
        n_samples = len(signal)
        identity_matrix = np.eye(n_samples)
        hartley_matrix = self._hartley_transform_matrix(n_samples)
        dictionary = np.hstack((identity_matrix, hartley_matrix))  # Joint dictionary [I, H]

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=50)
        omp.fit(dictionary, signal.reshape(-1, 1))
        sparse_code = omp.coef_.flatten()

        tc = dictionary @ sparse_code  # Reconstructed TC
        ssc = signal - tc  # SSC is the residual
        return tc, ssc

    def strong_tracking_kalman_filter(self, signal):
        n = len(signal)
        FF = np.ones(n)
        x_k = np.array([[signal[0]], [0], [0]])
        P_k = np.eye(3) * 0.1
        Q = np.eye(3) * 0.01
        R = np.array([[1]])
        F = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])
        H = np.array([[1, 0, 0]])
        lambda_k = 1
        beta = 2

        for k in range(1, n):
            x_k_pred = F @ x_k
            P_k_pred = F @ P_k @ F.T + Q
            epsilon = signal[k] - H @ x_k_pred
            S_k = H @ P_k_pred @ H.T + R
            K_k = P_k_pred @ H.T @ np.linalg.inv(S_k)
            zeta_k = max(epsilon.T @ np.linalg.inv(S_k) @ epsilon, 1e-8)
            lambda_k = beta / (beta + zeta_k)
            FF[k] = min(max(lambda_k, 0.5), 2.0)
            x_k = x_k_pred + K_k @ epsilon
            P_k = (np.eye(3) - K_k @ H) @ P_k_pred * lambda_k
        return FF

    def wavelet_threshold(self, signal, FF, wavelet="db4", level=3):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        thresholded_coeffs = []
        for i, c in enumerate(coeffs):
            if i == 0:
                thresholded_coeffs.append(c)
            else:
                FF_mean = np.mean(FF)
                alpha = 2 if FF_mean > 1 else 1
                sigma = np.median(np.abs(c)) / 0.6745
                threshold = alpha * sigma * np.sqrt(2 * np.log(len(c)))
                thresholded_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
        return thresholded_coeffs

    def process_steady_state_component(self, signal, threshold=0.05):
        fft_coeffs = fft(signal)
        magnitude = np.abs(fft_coeffs)
        indices = magnitude > threshold
        compressed_fft = fft_coeffs * indices
        non_zero_positions = np.where(indices)[0]
        non_zero_values = compressed_fft[non_zero_positions]
        return non_zero_positions, non_zero_values

    def reconstruct_steady_state_component(self, positions, values, n_samples):
        fft_coeffs = np.zeros(n_samples, dtype=complex)
        fft_coeffs[positions] = values
        reconstructed_signal = np.real(ifft(fft_coeffs))
        return reconstructed_signal

    def compress_and_analyze(self, signal_type):
        results = {}
        for snr in self.snr_levels:
            signal = self.generate_signal(signal_type, snr)
            tc, ssc = self.sparse_decomposition(signal)

            # Process TC with wavelet
            FF = self.strong_tracking_kalman_filter(tc)
            wavelet_coeffs = self.wavelet_threshold(tc, FF)

            # Process SSC with FFT zeroing
            positions, values = self.process_steady_state_component(ssc)

            # Reconstruction
            reconstructed_tc = pywt.waverec(wavelet_coeffs, wavelet="db4")
            reconstructed_ssc = self.reconstruct_steady_state_component(positions, values, len(signal))
            reconstructed_signal = reconstructed_tc + reconstructed_ssc

            # Metrics
            prd = np.sqrt(np.sum((signal - reconstructed_signal) ** 2) / np.sum(signal ** 2)) * 100
            cr = len(signal) * signal.itemsize * 8 / (len(wavelet_coeffs) + len(positions) + len(values))
            results[f"SNR_{snr}"] = {"original": signal, "reconstructed": reconstructed_signal, "metrics": {"prd": prd, "cr": cr}}
        return results

    def plot_results(self, signal_type, results):
        plt.figure(figsize=(18, 6))
        for i, (snr_label, result) in enumerate(results.items(), start=1):
            plt.subplot(1, len(results), i)
            plt.plot(result['original'], label='Original Signal', alpha=0.7)
            plt.plot(result['reconstructed'], label='Reconstructed Signal', linestyle='--')
            plt.title(f"{signal_type.replace('_', ' ').title()} - {snr_label}\n" +
                      f"CR: {result['metrics']['cr']:.2f}, PRD: {result['metrics']['prd']:.2f}%")
            plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    compressor = PQDSignalCompressor()

    signal_types = ["impulse", "sag", "decaying_harmonics"]
    for signal_type in signal_types:
        print(f"\nAnalyzing {signal_type.replace('_', ' ').title()} Signal:")
        results = compressor.compress_and_analyze(signal_type)
        compressor.plot_results(signal_type, results)
        for snr, metrics in results.items():
            print(f"  {snr}: Compression Ratio = {metrics['metrics']['cr']:.2f}, PRD = {metrics['metrics']['prd']:.2f}%")
