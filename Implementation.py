import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.fftpack import fft, ifft
from collections import Counter
import heapq

class PQDSignalCompressor:
    def __init__(self, fs=6400, duration=0.1, snr_levels=[30,40,50]):
        self.fs = fs
        self.duration = duration
        self.snr_levels = snr_levels

    def add_noise(self, signal, snr):
        power_signal = np.mean(signal ** 2)
        power_noise = power_signal / (10 ** (snr / 10))
        noise = np.sqrt(power_noise) * np.random.randn(len(signal))
        return signal + noise

    def generate_signal(self, signal_type, snr):
        t = np.arange(0, self.duration, 1/self.fs)

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

    def create_overcomplete_dictionary(self, n_samples):
        identity_matrix = np.eye(n_samples)
        hartley_matrix = self._hartley_transform_matrix(n_samples)
        return np.hstack((identity_matrix, hartley_matrix))

    def _hartley_transform_matrix(self, n_samples):
        indices = np.arange(n_samples)
        matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            matrix[i, :] = np.cos(2 * np.pi * i * indices / n_samples) + \
                           np.sin(2 * np.pi * i * indices / n_samples)
        return matrix / np.sqrt(n_samples)

    def sparse_decomposition(self, signal, dictionary, n_nonzero_coefs=50):
        signal = signal.reshape(-1, 1)
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=1e-3)
        omp.fit(dictionary, signal)
        return omp.coef_.flatten()

    def strong_tracking_kalman_filter(self, signal):
        n = len(signal)
        FF = np.ones(n)

        # Initialization
        x_k = np.array([[signal[0]], [0], [0]])  # State vector: [position; velocity; acceleration]
        P_k = np.eye(3) * 0.1  # Initial covariance matrix
        Q = np.eye(3) * 0.01   # Process noise covariance
        R = np.array([[1]])    # Measurement noise covariance
        F = np.array([[1, 1, 0.5],
                      [0, 1, 1],
                      [0, 0, 1]])  # State transition matrix
        H = np.array([[1, 0, 0]])  # Measurement matrix

        lambda_k = 1  # Forgetting factor
        beta = 2      # Tuning parameter

        for k in range(1, n):
            # Prediction
            x_k_pred = F @ x_k
            P_k_pred = F @ P_k @ F.T + Q

            # Compute the fading factor
            epsilon = signal[k] - H @ x_k_pred
            S_k = H @ P_k_pred @ H.T + R
            K_k = P_k_pred @ H.T @ np.linalg.inv(S_k)
            zeta_k = epsilon.T @ np.linalg.inv(S_k) @ epsilon
            lambda_k = beta / (beta + zeta_k)
            FF[k] = lambda_k

            # Update
            x_k = x_k_pred + K_k * epsilon
            P_k = (np.eye(3) - K_k @ H) @ P_k_pred * lambda_k

        return FF

    def wavelet_threshold(self, signal, FF, wavelet="db4", level=3):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        thresholded_coeffs = []

        for i, c in enumerate(coeffs):
            if i == 0:
                thresholded_coeffs.append(c)
            else:
                # Dynamic thresholding using FF
                FF_mean = np.mean(FF)
                alpha = 2 if FF_mean > 1 else 1
                sigma = np.median(np.abs(c)) / 0.6745
                threshold = alpha * sigma * np.sqrt(2 * np.log(len(c)))
                thresholded_coeffs.append(pywt.threshold(c, threshold, mode="soft"))

        return thresholded_coeffs

    def compress_transient_component(self, coeffs):
        # Flatten the coefficients
        flat_coeffs = np.hstack(coeffs)
        # Quantize coefficients
        quantized_coeffs = np.round(flat_coeffs * 1000).astype(int)
        # Huffman encoding
        freq = Counter(quantized_coeffs)
        huffman_codes = self.build_huffman_tree(freq)
        encoded_data = ''.join([huffman_codes[symbol] for symbol in quantized_coeffs])
        return encoded_data, huffman_codes

    def build_huffman_tree(self, freq):
        heap = [[weight, [symbol, '']] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huffman_codes = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
        return huffman_codes

    def process_steady_state_component(self, signal, threshold=0.01):
        # FFT of the steady-state component
        fft_coeffs = fft(signal)
        magnitude = np.abs(fft_coeffs)
        # Zeroing small coefficients
        indices = magnitude > threshold * np.max(magnitude)
        compressed_fft = fft_coeffs * indices
        # Store non-zero values and their positions
        non_zero_positions = np.where(indices)[0]
        non_zero_values = compressed_fft[non_zero_positions]
        return non_zero_positions, non_zero_values

    def reconstruct_steady_state_component(self, positions, values, n_samples):
        fft_coeffs = np.zeros(n_samples, dtype=complex)
        fft_coeffs[positions] = values
        reconstructed_signal = np.real(ifft(fft_coeffs))
        return reconstructed_signal

    def calculate_compression_metrics(self, original, compressed_data, reconstructed):
        # Compression Ratio
        original_size = original.size * original.itemsize * 8  # bits
        compressed_size = len(compressed_data)  # bits
        cr = original_size / compressed_size if compressed_size != 0 else np.inf

        # PRD
        prd = np.sqrt(np.sum((original - reconstructed) ** 2) / np.sum(original ** 2)) * 100

        return {
            'compression_ratio': cr,
            'prd': prd
        }

    def compress_and_analyze(self, signal_type):
        results = {}

        for snr in self.snr_levels:
            # Generate signal
            signal = self.generate_signal(signal_type, snr)

            # Create overcomplete dictionary
            n_samples = len(signal)
            dictionary = self.create_overcomplete_dictionary(n_samples)

            # Sparse decomposition
            sparse_code = self.sparse_decomposition(signal, dictionary)

            # Strong Tracking Kalman Filter
            FF = self.strong_tracking_kalman_filter(signal)

            # Wavelet thresholding with dynamic adjustment using FF
            compressed_coeffs = self.wavelet_threshold(sparse_code[:n_samples], FF)

            # Compress transient component using Huffman encoding
            compressed_data, huffman_codes = self.compress_transient_component(compressed_coeffs)

            # Process steady-state component
            positions, values = self.process_steady_state_component(signal)

            # Reconstruct steady-state component
            reconstructed_ssc = self.reconstruct_steady_state_component(positions, values, n_samples)

            # Reconstruct transient component
            reconstructed_tc = pywt.waverec(compressed_coeffs, wavelet="db4")

            # Combine components
            reconstructed_signal = reconstructed_ssc + reconstructed_tc

            # Calculate metrics
            metrics = self.calculate_compression_metrics(
                signal, compressed_data, reconstructed_signal
            )

            results[f"SNR_{snr}"] = {
                "original": signal,
                "compressed_data": compressed_data,
                "reconstructed": reconstructed_signal,
                "metrics": metrics
            }

        return results

# Main driver function
compressor = PQDSignalCompressor()

# Signal types to analyze
signal_types = ["impulse", "sag", "decaying_harmonics"]

# Comprehensive analysis
comprehensive_results = {}
for signal_type in signal_types:
    print(f"\nAnalyzing {signal_type.replace('_', ' ').title()} Signal:")
    comprehensive_results[signal_type] = compressor.compress_and_analyze(signal_type)

# Visualization
plt.figure(figsize=(18, 12))

for i, (signal_type, snr_results) in enumerate(comprehensive_results.items(), 1):
    for j, (snr_label, result) in enumerate(snr_results.items()):
        plt.subplot(len(signal_types), 3, (i-1)*3 + j + 1)

        plt.plot(result['original'], label='Original', alpha=0.7)
        plt.plot(result['reconstructed'], label='Reconstructed', linestyle='--')

        plt.title(f"{signal_type.replace('_', ' ').title()} - {snr_label}\n" +
                  f"CR: {result['metrics']['compression_ratio']:.2f}, " +
                  f"PRD: {result['metrics']['prd']:.2f}%")

        if j == 0:
            plt.ylabel(signal_type.replace('_', ' ').title())

        plt.legend()

plt.tight_layout()
plt.show()

# Print detailed metrics
print("\nDetailed Compression Analysis:")
for signal_type, snr_results in comprehensive_results.items():
    print(f"\n{signal_type.replace('_', ' ').title()} Signal:")
    for snr_label, result in snr_results.items():
        print(f"  {snr_label}:")
        for metric, value in result['metrics'].items():
            print(f"    {metric.replace('_', ' ').title()}: {value}")