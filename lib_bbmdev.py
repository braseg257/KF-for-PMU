import pandas as pd
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
from tqdm import tqdm

import os

def layout_coil_current_voltage(ax, xlabel, ylabel):
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_multiple_arrays_with_time_and_color(
    ax, time_data, data_arrays, labels=None, plot_col=None, alpha = 1
):
    """
    Plot multiple data arrays with a common time axis on the same axes.

    Parameters:
    - ax (matplotlib.axes._axes.Axes): Axes object to plot on.
    - time_data: Common time data for the x-axis.
    - data_arrays (list): List of data arrays to be plotted on the y-axis.
    - labels (list): List of labels for each data array. (Optional)
    - plot_col (list): List of colors for each data array. If None, default colors are used. (Optional)
    """
    labels = labels or [f"Data {i+1}" for i in range(len(data_arrays))]
    plot_col = plot_col or ["C{}".format(i) for i in range(len(data_arrays))]

    for data, label, color in zip(data_arrays, labels, plot_col):
        ax.plot(time_data, data, label=label, color=color, alpha = alpha)

def create_directory(filename, datetime = False):
    # Get the current date
    if datetime:
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")

    # Create a new directory name
        directory_name = f"{current_datetime}_{filename}"
    else:
        directory_name = f"{filename}"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' has been created.")
    else:
        print(f"Directory '{directory_name}' already exists.")

    return directory_name

def medianfilter1D(arr, window_size):
    """
    Apply a 1D median filter to the input array.

    Parameters:
    - arr (numpy.ndarray): The input 1D array.
    - window_size (int): The size of the window for the median filter.

    Returns:
    - numpy.ndarray: The result of the median filter.
    """

    # Calculate the size of the padding needed
    array_length = len(arr)
    index_size = window_size//2
    #print(index_size)

    # Generate the column indices for the rolling window
    index_column = np.arange(0 - (index_size), len(arr) - index_size, 1)

    # Create the 2D array of indices for the rolling window
    index_array = index_column + np.arange(0, window_size)[:, np.newaxis]
    #print(index_array)

    # Apply padding by clipping indices to valid range
    index_array[index_array < 0] = 0
    index_array[index_array > array_length - 1] = array_length - 1

    # Calculate median along the specified axis
    result = np.median(arr[index_array], axis=0)
    return result

def fft_specgram(y_arr, N, dt, plot = False):
    """
    Compute the frequency, amplitude, and phase angle (in degrees) components of the FFT of a given signal.

    Parameters:
        y_arr (array_like): Input signal array.
        N (int): Number of data points in the signal.
        dt (float): Time step between consecutive data points in the signal.
        plot (bool, optional): Whether to plot the FFT components. Default is False.

    Returns:
        tuple: A tuple containing:
            y_fft_nyq (ndarray): FFT components up to Nyquist frequency.
            freq_nyq (ndarray): Array of frequency values corresponding to the FFT components up to Nyquist frequency.
            Amp_arr_nyq (ndarray): Array of amplitudes of the FFT components up to Nyquist frequency.
            Deg_arr_nyq (ndarray): Array of phase angles (in degrees) of the FFT components up to Nyquist frequency.
    """
    # Compute the FFT of the input signal
    y_fft = np.fft.fft(y_arr)
    y_fft_nyq = [buf[1:int(N / 2)] for buf in y_fft]

    # Compute the frequencies corresponding to the FFT components
    freq = np.fft.fftfreq(N, d=dt)
    freq_nyq = freq[1:int(N / 2)]

    # Compute the amplitude of each FFT component
    Amp_arr = abs(y_fft / (N / 2))
    Amp_arr_nyq = [buf[1:int(N / 2)] for buf in Amp_arr]

    # Compute the phase angle (in degrees) of each FFT component
    Deg_arr = np.angle(y_fft, deg=True)
    Deg_arr_nyq = [buf[1:int(N / 2)] for buf in Deg_arr]

    if plot == False:
        pass
    else:
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(15, 9))
        # グリッドの設定 (3行, 1列)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.0)

        # Frequency domain magnitude
        ax0 = plt.subplot(gs[0])
        ax0.stem(freq_nyq, abs(y_fft_nyq[0]),label='Frequency')
        ax0.legend()
        ax0.set_ylabel('Frequency domain magnitude')
        #ax0.set_ylim(0, max(y_fft[1:int(N / 2)])+1)

        # Amplitude
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.stem(freq_nyq, Amp_arr_nyq[0],label='Amplitude')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.set_ylabel('Signal Amplitude')

        # Phase
        ax2 = plt.subplot(gs[2], sharex=ax0)
        ax2.stem(freq_nyq, Deg_arr_nyq[0],label='Phase')
        ax2.legend()
        ax2.set_xscale('log')
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [Hz]')

        plt.show()

    return y_fft_nyq,freq_nyq, Amp_arr_nyq, Deg_arr_nyq


def fft_argmax(freqs, Amp_arr, Deg_arr, argmax=None, print_output=False):
    if argmax is None:
        argmax = np.abs(Amp_arr).argmax()
        frequency = freqs[argmax]
        amp = Amp_arr[argmax]
        deg = Deg_arr[argmax]
    else:
        frequency = freqs[argmax]
        amp = Amp_arr[argmax]
        deg = Deg_arr[argmax]

    if not print_output:
        pass
    else:
        print({"frequency": frequency, "amplitude": amp, "degree": deg})

    return argmax, frequency, amp, deg


def tdms_load_selected_custom(file_name, data_dict=None, samplerate=1000, print_=True):
    if data_dict == None:
        data_dict = {'cDAQ1Mod1/ai0':'Iu','cDAQ1Mod1/ai1':'Iv',  'cDAQ1Mod1/ai2':'Iw', 
             'cDAQ1Mod2/ai0':'A', 'cDAQ1Mod2/ai1':'B', 'cDAQ1Mod2/ai2':'Z', 'cDAQ1Mod2/ai3':'H',
             'cDAQ1Mod3/ai0':"SV1", 'cDAQ1Mod3/ai1':"SV2", 'cDAQ1Mod3/ai2':"SV3",'cDAQ1Mod3/ai3':"Count",
             'cDAQ1Mod4/ai0':"Vu", 'cDAQ1Mod4/ai1':"Vv", 'cDAQ1Mod4/ai2':'Vw'}
    else:
        pass
    tdms_file = TdmsFile(file_name)
    group = tdms_file.groups()[0]
    data_lab = ['time']
    if print_:
        count = 0
        for channel in group.channels():
            channel_name = channel.name
            #print("channel:", channel_name, "index:", count)
            count += 1
            if channel_name in data_dict.keys():
                data_lab.append(data_dict[channel_name])
    selected_arr=np.ones(len(group.channels()))
    selected_read_data = []
    for j in range(0, len(group.channels())):
        if selected_arr[j] == 1:
            selected_read_data.append(group.channels()[j])
        else:
            # nanimo sinai
            pass
    # for i in tqdm(range(0, len(selected_read_data)), desc="Data loading"):
    for i in range(0, len(selected_read_data)):
        channel = selected_read_data[i]
        data = channel[:]
        if i == 0:
            data_arr = np.empty([len(selected_read_data) + 1, len(data)])
            data_arr[0] = np.linspace(
                0, len(data) / samplerate, len(data), endpoint=False
            )
            data_arr[i + 1] = data
        else:
            data_arr[i + 1] = data
    df = pd.DataFrame(data_arr, index=data_lab)
    return df.T

def tdms_load_selected(file_name, data_lab, selected_arr, samplerate=1000, print_=True):
    tdms_file = TdmsFile(file_name)
    group = tdms_file.groups()[0]
    if print_:
        count = 0
        for channel in group.channels():
            channel_name = channel.name
            print("channel:", channel_name, "index:", count)
            count += 1

    selected_read_data = []
    for j in range(0, len(group.channels())):
        if selected_arr[j] == 1:
            selected_read_data.append(group.channels()[j])
        else:
            # nanimo sinai
            pass
    # for i in tqdm(range(0, len(selected_read_data)), desc="Data loading"):
    for i in range(0, len(selected_read_data)):
        channel = selected_read_data[i]
        data = channel[:]
        if i == 0:
            data_arr = np.empty([len(selected_read_data) + 1, len(data)])
            data_arr[0] = np.linspace(
                0, len(data) / samplerate, len(data), endpoint=False
            )
            data_arr[i + 1] = data
        else:
            data_arr[i + 1] = data
    df = pd.DataFrame(data_arr, index=data_lab)
    return df.T


def time_crossing_regression(signal, time, threshold=None):
    """
    Function to find the times when a given signal crosses a specified threshold.

    Parameters:
    - signal: numpy array
        The signal to be analyzed.
    - time: numpy array
        Time values corresponding to the signal.
    - threshold: float, optional
        The threshold used as the reference axis. Default is the average of the signal's maximum and minimum values.

    Returns:
    - a: numpy array
        Array of slope values for the lines fitted to the threshold crossings.
    - b: numpy array
        Array of intercept values for the lines fitted to the threshold crossings.
    - time_stamp: numpy array
        Array of timestamps where the signal crosses the low threshold.
    """
    # Set default value for the threshold
    if threshold is None:
        threshold = (np.max(signal) + np.min(signal)) / 2

    # Create a binary array indicating whether the signal crosses the threshold
    binary_signal = signal > threshold

    # Convert the boolean values to integers (True → 1, False → 0)
    binary_signal_int = binary_signal.astype(int)

    # Calculate the differences to next element of binary_signal_int [0,1,1,1,1,1,0,1] -> [1,0,0,0,0,0,1,0]
    # we can get both tachiagari and tachisagari.
    signal_diff = np.diff(binary_signal_int)

    # Get the indices where the difference changes from 1 to 0
    crossing_indices = np.where(signal_diff)[0]

    # Fit a line to the two points around the threshold crossing
    a = (signal[crossing_indices + 1] - signal[crossing_indices]) / (
        time[crossing_indices + 1] - time[crossing_indices]
    )
    b = signal[crossing_indices] - a * time[crossing_indices]

    # Calculate the timestamp where the data crosses the low threshold
    time_stamp = ([threshold] * len(b) - b) / a

    # Return the fitted line parameters and timestamps
    return threshold, time_stamp, crossing_indices


def triggerd_low_threshold_crossings(signal, time, high_threshold, low_threshold):
    time_stamps = np.array([])
    th, ts, high_crossing_indices = time_crossing_regression(
        signal, time, high_threshold
    )
    for i in range(0, len(high_crossing_indices[1::2])):
        # Extract surrounding data around the high threshold crossing
        window_size: int = 500
        start_index = high_crossing_indices[1::2][i] - window_size
        end_index = high_crossing_indices[1::2][i] + window_size

        chunk_signal = signal[start_index:end_index]
        chunk_time = time[start_index:end_index]

        # Create a binary array indicating whether the signal crosses the low threshold
        binary_signal = chunk_signal > low_threshold
        binary_signal_int = binary_signal.astype(int)

        # Calculate the differences at moments when the signal crosses the threshold
        signal_diff = np.diff(binary_signal_int)

        # Get the indices where the difference changes from 1 to 0
        crossing_indices = np.where(signal_diff)[0]

        # Fit a line to the two points around the threshold crossing
        a = (chunk_signal[crossing_indices + 1] - chunk_signal[crossing_indices]) / (
            chunk_time[crossing_indices + 1] - chunk_time[crossing_indices]
        )
        b = chunk_signal[crossing_indices] - a * chunk_time[crossing_indices]

        # Calculate the timestamp where the data crosses the low threshold
        time_stamp = ([low_threshold] * len(b) - b) / a
        time_stamps = np.append(time_stamps, time_stamp)

    return low_threshold, time_stamps


def triggerd_low_threshold_crossings_skipsec(
    signal, time, high_threshold, low_threshold, skip_sec, plot=False
):
    time_stamps = np.array([])
    th, ts, high_crossing_indices = time_crossing_regression(
        signal, time, high_threshold
    )
    time_stamps_diff = np.diff(ts)
    # print(time_stamps_diff)
    # print(len(ts),len(high_crossing_indices))
    # print("before cut",high_crossing_indices)
    time_stamps_indices = np.where(time_stamps_diff > skip_sec)[0]
    high_crossing_indices_2 = high_crossing_indices[time_stamps_indices]
    # print("after cut",high_crossing_indices_2)

    for i in range(0, len(high_crossing_indices_2)):
        # Extract surrounding data around the high threshold crossing
        window_size: int = 500
        start_index = high_crossing_indices_2[i] - window_size
        end_index = high_crossing_indices_2[i] + window_size

        chunk_signal = signal[start_index:end_index]
        chunk_time = time[start_index:end_index]
        if plot:
            # if i % 20 == 0:
            plt.close()
            plt.plot(chunk_time, chunk_signal)
            plt.show()

        # Create a binary array indicating whether the signal crosses the low threshold
        binary_signal = chunk_signal > low_threshold
        binary_signal_int = binary_signal.astype(int)

        # Calculate the differences at moments when the signal crosses the threshold
        signal_diff = np.diff(binary_signal_int)

        # Get the indices where the difference changes from 1 to 0
        crossing_indices = np.where(signal_diff)[0]
        # Fit a line to the two points around the threshold crossing
        a = (chunk_signal[crossing_indices + 1] - chunk_signal[crossing_indices]) / (
            chunk_time[crossing_indices + 1] - chunk_time[crossing_indices]
        )
        b = chunk_signal[crossing_indices] - a * chunk_time[crossing_indices]

        # Calculate the timestamp where the data crosses the low threshold
        time_stamp = ([low_threshold] * len(b) - b) / a
        time_stamps = np.append(time_stamps, time_stamp)

    time_stamps_diff = np.diff(time_stamps)
    # print(time_stamps_diff)
    # print(len(ts),len(high_crossing_indices))
    # print("before cut",time_stamps)
    time_stamps_indices = np.where(time_stamps_diff > skip_sec)[0]
    # time_stamps_indices = np.append(np.array([0]),time_stamps_indices+1)
    time_stamps_indices = time_stamps_indices + 1
    time_stamps_selected = time_stamps[time_stamps_indices]
    # print("after cut",time_stamps_selected)

    return low_threshold, time_stamps_selected


def time2freq(time_stamp):
    period = np.diff(time_stamp)
    frequency = 1 / np.asarray(period)
    return period, frequency


def timediff2phase(time_stamp1, time_stamp2, return_degrees=True):
    """
    Calculate the phase difference between two sets of time stamps.

    Parameters:
    - time_stamp1: numpy array
        The first set of time stamps.
    - time_stamp2: numpy array
        The second set of time stamps.
    - return_degrees: bool, optional
    If True, the results are returned in degrees. If False, results are returned in radians.
    Default is True.

    Returns:
    - phase_ave: float
        The average phase difference in degrees.
    - phase_arr: numpy array
        Array of individual phase differences in degrees.
    """
    # Calculate the average time interval between consecutive time stamps
    dt = np.average(time_stamp1[1:] - time_stamp1[:-1])

    # Determine the target length as the minimum of the lengths of the two arrays
    target_length = min(len(time_stamp1), len(time_stamp2))

    # Trim both arrays to the shorter length
    trimmed_time_stamp1 = time_stamp1[:target_length]
    trimmed_time_stamp2 = time_stamp2[:target_length]

    # Calculate the absolute phase differences and convert to degrees
    # omega = 2.0*np.pi/dt
    phase_arr = np.abs(trimmed_time_stamp2 - trimmed_time_stamp1) * 1.0 / dt

    if return_degrees == True:
        phase_arr *= 180.0
    else:
        phase_arr *= np.pi

    # Calculate the average phase difference
    phase_ave = np.average(phase_arr)

    return phase_ave, phase_arr


def triggerd_window(
    signal, time, high_threshold, low_threshold, window_size: int, plot=False
):
    # Initialize an empty array to store timestamps
    time_stamps = np.array([])

    # Find the time and index of high threshold crossings
    th, ts, high_crossing_indices = time_crossing_regression(
        signal, time, high_threshold
    )

    # Loop over each high threshold crossing
    for i in range(0, len(high_crossing_indices) - 1):
        # Extract the signal and time window centered around the high threshold crossing
        start_index = high_crossing_indices[1:][i] - window_size
        end_index = high_crossing_indices[1:][i] + window_size
        chunk_signal = signal[start_index:end_index]
        chunk_time = time[start_index:end_index]

        # Calculate the gradient of the chunk
        chunk_grad = chunk_signal[-1] - chunk_signal[0]

        # Plot the chunk if plot flag is True
        if plot:
            plt.close()
            plt.plot(chunk_time, chunk_signal)
            plt.show()

        # Check if the gradient is positive
        if chunk_grad >= 0:
            # Create a binary array indicating whether the signal crosses the low threshold
            binary_signal = chunk_signal > low_threshold
            binary_signal_int = binary_signal.astype(int)

            # Calculate the differences at moments when the signal crosses the threshold
            signal_diff = np.diff(binary_signal_int)

            # Get the indices where the difference changes from 1 to 0
            crossing_indices = np.where(signal_diff)[0]

            # Fit a line to the two points around the threshold crossing
            a = (
                chunk_signal[crossing_indices + 1] - chunk_signal[crossing_indices]
            ) / (chunk_time[crossing_indices + 1] - chunk_time[crossing_indices])
            b = chunk_signal[crossing_indices] - a * chunk_time[crossing_indices]

            # Calculate the timestamp where the data crosses the low threshold
            time_stamp = ([low_threshold] * len(b) - b) / a

            # Append the timestamps to the array
            time_stamps = np.append(time_stamps, time_stamp)

    len_unique = np.unique(time_stamps).size
    len_stamps = time_stamps.size

    if len_unique == len_stamps:
        pass
    else:
        errors: str = (
            "The lengths of natural timestamps and unique keys differ. This is due to duplicate timestamps passing through the threshold. Please set a small window."
        )
        print(errors)

    # Return the low threshold and the calculated timestamps
    return low_threshold, time_stamps


def freq_calc(time, data, thre_H, thre_L):
    #amp = max(data) - min(data)
    #thre_H = np.mean(data) + 0.5*np.std(data)
    #thre_L = np.mean(data) - 0.5*np.std(data)
    switch = 0
    time_stamp_arr = []
    time_diff = []
    time_index = []
    initial_time_switch = True
    for i in tqdm(range(0, len(time)-1),desc = 'Calculating rotation frequency...'):
        if data[i] < thre_H and data[i+1] >= thre_H \
        and switch == 0:
            switch = 1
        elif data[i] > thre_L and data[i+1] <= thre_L \
        and switch == 1:
            switch = 0
            a = float(data[i+1] - data[i])/(time[i+1] - time[i])
            b = data[i] - a*time[i]
            time_stamp = float(thre_L - b)/a
            time_stamp_arr.append(time_stamp)
            time_index.append(i)
            #time_diff.append(abs((time_stamp-time[i])/(time[i+1]-time[i])))
    period = []
    period = [time_stamp_arr[i+1] - time_stamp_arr[i] for i in range(len(time_stamp_arr)-1)]
    period_accum = np.cumsum(np.array(period)) + time_stamp_arr[0]
    return period_accum, period, time_stamp_arr, time_index

def freq_calc_deriv(time, signal, high_TH,low_TH, window=0):
    maxim = []
    minim = []
    time_index = []
    sign = np.sign((signal[1]- signal[0])/(time[1]-time[0])) #sign of the derivative of first point
    count = -1*window
    for i in range(1, len(time)-1):
        deltat = time[i+1]-time[i]
        deltas = signal[i+1]- signal[i]
        deriv = deltas/deltat
        #if deltas ==0:
           #print(i)
        if int(np.sign(deriv)*sign) == -1 and np.sign(deriv)==-1 and signal[i]>=high_TH:
          if i >= count + window:
            maxim.append(time[i])
            time_index.append(i)
          count = i
        if int(np.sign(deriv)*sign) == -1 and np.sign(deriv)==+1 and signal[i]<=low_TH:
          minim.append(time[i])
        if np.sign(deriv)!=0:
          sign = np.sign(deriv)
    period = [maxim[i+1] - maxim[i] for i in range(len(maxim)-1)] 
    period_accum = np.cumsum(np.array(period)) + maxim[0]
    return period_accum, period, maxim, time_index

def stability_zone_start_end(time, start, end, print_ = True):
    """
    Returns the indices of time input array corresponding to start and end.

    Parameters:
        time (array_like): Input signal array.
        start (float): Start of the stability zone.
        end (float): End of the stability zone.
    Returns:
        tuple: A tuple containing:
            stab_start: Index of time such that time[stab_start] = start +- approx. error
            stab_end: Index of time such that time[stab_end] = end +- approx. error
    """

    stab_start = np.where(time>=start)[0][0]
    stab_end = np.where(time<=end)[0][-1]
    if print_==True:
        print(f'Stability zone goes from index {stab_start} to index {stab_end}.')
    return stab_start, stab_end


def stability_zone_zoom(signal, time, start, end, margin = 0.1, print_=True):
    """
    Returns the indices of time input array corresponding to start and end.

    Parameters:
        signal (list of arrays): Input signal of same length to zoom around the stability zone.
        time (array_like): Input common time signal array.
        start (float): Start of the stability zone.
        end (float): End of the stability zone.
        zoom (float): Percentage of margin above maximum and below minimum (default=0.1)
    Returns:
        array: A (len(signal), 2)-array containing the lower and upper y values for each signal

    """
    zooms = []
    signal = np.array(signal)
    stab_start, stab_end = stability_zone_start_end(time, start, end, print_)
    
    if len(signal.shape)==1:
        if len(signal)==len(time):
            signal_trim = signal[stab_start:stab_end]
            y_min, y_max = np.min(signal_trim), np.max(signal_trim)
            zooms.append([(1-margin*np.sign(y_min))*y_min, (1+margin*np.sign(y_max))*y_max])
        else:
            errors: str = (
            f"The length of input signal 0 is different from the length of time signal ({len(signal)} =! {len(time)}), maybe the sampling rate is different?"
            )
            raise ValueError(errors)
    else:             
        for elem in signal: 
            if len(elem)==len(time): #check for mismatches in the sampling time
                signal_trim = elem[stab_start:stab_end]
                y_min, y_max = np.min(signal_trim), np.max(signal_trim)
                print(y_min, y_max)
                zooms.append([(1-margin*np.sign(y_min))*y_min, (1+margin*np.sign(y_max))*y_max])
                
            else:
                errors: str = (
                    f"The length of input signal {np.where(s==elem for s in signal)[0][0]} is different from the length of time signal ({len(elem)} =! {len(time)}), maybe the sampling rate is different?"
                )
                raise ValueError(errors)
    if print_ == True:
        print(zooms)  
    return np.array(zooms)


def Cut_start(df_signal, thre, samplerate):
    try:
        thre_indv = np.where(abs(df_signal["Iv"]) > thre)[0][0]
        thre_indu = np.where(abs(df_signal["Iu"]) > thre)[0][0]
        thre_indw = np.where(abs(df_signal["Iw"]) > thre)[0][0]

        if thre_indv == thre_indu == thre_indw:
            skip1 = thre_indu + 90 * samplerate

            thre_indv = np.where(abs(df_signal["Iv"][skip1:]) > thre)[0][0]
            thre_indu = np.where(abs(df_signal["Iu"][skip1:]) > thre)[0][0]
            thre_indw = np.where(abs(df_signal["Iw"][skip1:]) > thre)[0][0]

            if thre_indv == thre_indu == thre_indw:
                cut_time = df_signal["time"][thre_indv + skip1]
                print(cut_time)
            else:
                print(
                    "Second threshold error,Tu, Tv, Tw = ",
                    df_signal["time"][thre_indu + skip1],
                    df_signal["time"][thre_indv + skip1],
                    df_signal["time"][thre_indw + skip1],
                )
        else:
            print(
                "First threshold error, Tu, Tv, Tw = ",
                df_signal["time"][thre_indu],
                df_signal["time"][thre_indv],
                df_signal["time"][thre_indw],
            )

        df_new = df_signal.loc[thre_indu + skip1 :]

    except Exception as e:
        print(e)
        return e
    return df_new, cut_time


def Cut_start2(df_signal, thre, samplerate):
    try:
        thre_indv = np.where(abs(df_signal["Iv"]) > thre)[0][0]
        thre_indu = np.where(abs(df_signal["Iu"]) > thre)[0][0]
        thre_indw = np.where(abs(df_signal["Iw"]) > thre)[0][0]
        
        if thre_indv == thre_indu == thre_indw:
            skip1 = thre_indu + 90 * samplerate
            cut_time = df_signal["time"][thre_indv + skip1]
            print(cut_time)

        else:
            print(
                "First threshold error, Tu, Tv, Tw = ",
                df_signal["time"][thre_indu],
                df_signal["time"][thre_indv],
                df_signal["time"][thre_indw],
            )
        
        df_new = df_signal.loc[thre_indu + skip1 :]

    except Exception as e:
        print(e)
        return e
    return df_new, cut_time
