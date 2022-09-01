import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float64)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END OF EX2 GIVE HEPLER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#  q1.1
def DFT(signal):
    """
    named definitive furrier transform (not fft), and preforms exactly that
    :param signal:
    :return:
    """
    dim = signal.shape[0]
    arr = signal.reshape(signal.shape[0])

    # exp of outer product([0,1...N-1],[0,1...N-1]) multiply by -2pi i
    coefficients = np.outer(np.linspace(0, dim - 1, dim), np.linspace(0, dim - 1, dim))
    coefficients = coefficients * (-2 * np.pi * 1j) / dim
    coefficients = np.exp(coefficients)

    result = np.dot(coefficients, arr.T).T

    if result.shape == signal.shape:
        return result
    else:
        return np.array([result], dtype=np.complex128).T


def IDFT(fourier_signal):
    """
    inverse definitive furrier transform (not fft) , and preforms exactly that
    :param fourier_signal: given complex128 np.array
    :return: return complex128 np array
    """
    dim = fourier_signal.shape[0]
    arr = fourier_signal.reshape(fourier_signal.shape[0])

    # exp of outer product([0,1...N-1],[0,1...N-1]) multiply by -2pi i
    coefficients = np.outer(np.linspace(0, dim - 1, dim), np.linspace(0, dim - 1, dim))
    coefficients = coefficients * (2 * np.pi * 1j) / dim
    coefficients = np.exp(coefficients)

    result = np.dot(coefficients, arr.T).T /dim

    if result.shape == fourier_signal.shape:
        return result
    else:
        return np.array([result], dtype=np.complex128).T


#  q1.2
def DFT2(image):
    """
    the same as DFT but this time works on and NxM np array
    :return: NxM complex array fitting the TF
    """

    # run ft on col by inverse in the img then for row in img = dft(row) <=> img[i] = DFT(img[i])
    image = np.apply_along_axis(DFT, 1, image)
    image = np.apply_along_axis(DFT, 0, image)

    return image


def IDFT2(signal):
    """
    the same as IDFT but this time works on and NxN np array
    :return: NxM complex array fitting the TF
    """

    # run ft on col by inverse in the img then for row in img = dft(row) <=> img[i] = DFT(img[i])
    signal = np.apply_along_axis(DFT, 1, signal)
    signal = np.apply_along_axis(DFT, 0, signal)

    return signal / (signal.shape[0] * signal.shape[1])


#  q2.1
def change_rate(filename, ratio):
    """
    Changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header. When the audio player uses the same samples as if they were taken
    in a higher sample rate, a “fast forward” effect is created. Given a WAV file, this function saves the audio
    in a new file called change_rate.wav. You can use the functions read, write from scipy.io.wavfile.
    The function should have the following interface:
    change_rate(filename, ratio)

    :param filename: filename is a string representing the path to a WAV file
    :param ration: ratio is a positive float64 representing the duration change.

    assumption: 0.25 < ratio < 4 .

    example: if the original sample rate is 4,000Hz and ratio is 1.25, then the new sample rate will be
    5,000Hz. The function should not return anything

    """
    sr, data = wavfile.read(filename)
    new_sr = sr * ratio
    wavfile.write('change_rate.wav', int(new_sr), data)


#  q2.2
def change_samples(filename, ratio):
    """

    :param filename:a string representing the path to a WAV file
    :param ratio:is a positive float64 representing the duration chang

    :return The function should return a 1D ndarray of dtype float64 representing the new sample points. You may
    assume that 0.25 < ratio < 4.

    """
    sr, data = wavfile.read(filename)
    new_data = resize(data, ratio)
    wavfile.write('change_samples.wav', sr, new_data)


def resize(data, ratio):
    """
    take data, do DFT(data), make the result symmetric over 0, then clip N/2 (number of samples to remove) from the sides of the result. convert back to
    :param data:
    :param ratio:
    :return:
    """

    #  change the signal (add or remove stuff from it)
    signal = np.fft.fftshift(DFT(data))  # create complex signal centered at zero

    if ratio > 1:
        signal = accelerate_resize(signal, ratio)
    elif ratio < 1:
        signal = decelerate_resize(signal, ratio)

    signal = np.fft.ifftshift(signal)  # return it to its original representation
    new_data = IDFT(signal)

    if data.dtype != np.complex128:
        new_data = np.real(new_data).T

    return new_data


def resize_spectrogram(data, ratio):
    """
    apple resize on the rows of te spectrogram indifferently.
    :param data:
    :param ratio:
    :return:
    """
    spectrogram = stft(data)
    length = len(resize(spectrogram[0], ratio))
    resized_spectrogram = np.zeros(shape=(spectrogram.shape[0], length), dtype=np.complex128)
    for row in range(spectrogram.shape[0]):
        resized_spectrogram[row] = resize(spectrogram[row], ratio)

    return np.real(istft(resized_spectrogram))


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram.
    You may assume that 0.25 < ratio < 4
    :param data: is a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: is a positive float64 representing the rate change of the WAV file
    :return:The function should return the given data
            rescaled according to ratio with the same datatype as data.
    """
    return np.real(istft(phase_vocoder(stft(data), ratio)))


# q3.1
def conv_der(image):
    """
    derive [0.5,0,-0.5] bth at x and y axis then computes the magnitude
    :param image:
    :return:
    """
    deriv_x = np.array([[0, 0, 0],
                        [0.5, 0, -0.5],
                        [0, 0, 0]])
    deriv_y = np.array([[0, 0.5, 0],
                        [0, 0, 0],
                        [0, -0.5, 0]])
    x_wise = np.abs(signal.convolve2d(image, deriv_x, mode='same', boundary='fill', fillvalue=0))
    y_wise = np.abs(signal.convolve2d(image, deriv_y, mode='same', boundary='fill', fillvalue=0))
    return np.square(x_wise**2 + y_wise**2)


# q3.2
def fourier_der(image):

    singals = DFT2(image)
    furrier_img_x = np.fft.fftshift(singals)  # to furrier, with (0,0) at the center
    furrier_img_y = np.fft.fftshift(singals.T)

    # dx = get_derivative_x_axis(image)

    # creates [[-N/2,-N+1/2,...,N/2],[-N/2,-N+1/2,...,N/2],...[-N/2,-N+1/2,...,N/2]] (N is the length of the rows)
    length, depth = image.shape[1], image.shape[0]
    grid_x = np.linspace(0, length-1, length)-int(length/2)
    grid_y = np.linspace(0, depth-1, depth)-int(depth/2)
    # matrix_x = np.tile(grid_x, (furrier_img.shape[0], 1))

    # multiply the the furrier img with the matrix_x (1 cell by 1 cell, not matrix multiplication), and by 2*pi*i/N
    X_furrier_img = furrier_img_x * grid_x * (2 * np.pi * 1j / length)
    Y_furrier_img = furrier_img_y * grid_y * (2 * np.pi * 1j / depth)

    # reversing the shift and the tf
    dx = np.abs(IDFT2(np.fft.ifftshift(X_furrier_img)))**2
    dy = np.abs(IDFT2(np.fft.ifftshift(Y_furrier_img)).T)**2

    return np.sqrt(dy + dx)


def get_derivative_x_axis(image):

    furrier_img = np.fft.fftshift(DFT2(image))  # to furrier, with (0,0) at the center

    length = furrier_img.shape[1] # creates [[-N/2,-N+1/2,...,N/2],[-N/2,-N+1/2,...,N/2],...[-N/2,-N+1/2,...,N/2]] (N is the length of the rows)
    grid_x = np.linspace(-int(length/2), -int(length/2), length)
    # matrix_x = np.tile(grid_x, (furrier_img.shape[0], 1))

    # multiply the the furrier img with the matrix_x (1 cell by 1 cell, not matrix multiplication), and by 2*pi*i/N
    X_furrier_img = furrier_img * grid_x * (2 * np.pi * 1j / length)

    # returns to the middle using, then returns to pixels by IDFT2
    dx = IDFT2(np.fft.ifftshift(X_furrier_img))
    return dx


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def accelerate_resize(signal, ratio):
    """
    sub function of resize:
    cropping the signal by its outer edges, to be smaller by ratio ratio.

    if crop is odd, ex[-2,-1,0,1,2]. crop 3: signal[floor(3/2):len(sig)-ciel(3/2)] = signal[1:3] = [-1,0]
    else          , ex[-2,-1,0,1,2]. crop 2: signal[floor(2/2):len(sig)-ciel(2/2)] = signal[1:4] = [-1,0,1]

    :param signal: furrier transformed signal
    :param ratio: float
    :return: cropped signal
    """
    overall_cilp_from_signal = len(signal) - int(np.floor(len(signal) / ratio))
    signal = signal[
             int(np.floor(overall_cilp_from_signal / 2)):len(signal) - int(np.ceil(overall_cilp_from_signal / 2))]
    return signal


def decelerate_resize(signal, ratio):
    """
    similar to accelerate_resize, but for slowing down.
    :param signal:
    :param ratio:
    :return:
    """
    add_to_each_side = abs(int(len(signal)/ratio)-len(signal))
    left = np.zeros(int(np.floor(add_to_each_side / 2)), dtype=signal.dtype)
    right = np.zeros(int(np.ceil(add_to_each_side / 2)), dtype=signal.dtype)
    signal = np.concatenate((left, signal, right))
    return signal


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing site ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import imageio as imao
from skimage import color


def read_image(filename, representation=2):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
                image (1) or an RGB image (2)
    :return: This function returns an image, make sure the output image is represented by a matrix of type
             np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    """
    img = imao.imread(filename)
    if representation == 1:
        img = color.rgb2gray(img)
    else:
        img = img / 256
    return np.float64(img)