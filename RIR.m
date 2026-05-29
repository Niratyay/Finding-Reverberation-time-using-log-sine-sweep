% --- STEP 1: Generate Signals ---
Fs = 48000;             % Sampling frequency (Hz)
T = 3;                  % Sweep duration (seconds)
f1 = 20;                % Start frequency (Hz)
f2 = 20000;             % End frequency (Hz)

% Time vector
t = (0:1/Fs:T)';

% Calculate L based on your notes
L = T / log(f2/f1);

% Forward Log Sine Sweep: x(t)
x = sin(2*pi*f1*L*(exp(t/L) - 1));

% Inverse Filter: x_inv(t)
A_t = exp(-t/L);

% Time reversal of x(t) is x(T-t)
x_rev = flip(x);

% Final inverse filter
x_inv = A_t .* x_rev;

% Save forward sweep to file 
audiowrite('sweep.wav', x, Fs);
disp('Sweep generated and saved.');


% --- STEP 2: Playback and Recording ---
% Setup audio recorder (Fs, 16-bit, 1 channel)
recObj = audiorecorder(Fs, 16, 1);

disp('Starting recording and playback in 2 seconds...');
pause(2);                 % Brief pause to let you quiet down

record(recObj);           % Start recording
sound(x, Fs);             % Play sweep through default output

% Pause for sweep duration + 2 seconds to capture the room's reverb tail
pause(T + 2);             

stop(recObj);             % Stop recording
y = getaudiodata(recObj); % Extract recorded array into variable y
disp('Recording complete.');


% --- STEP 3: Frequency Domain Deconvolution ---
% Note: You can replace the padding logic here with your adjustseq code, 
% but standard zero-padding via FFT length (N) ensures linear convolution.

% Find optimal FFT length (N) to prevent circular wrap-around
N = 2^nextpow2(length(y) + length(x_inv) - 1);

% Compute FFTs with zero-padding to length N
Y_k = fft(y, N);
X_inv_k = fft(x_inv, N);

% Frequency domain multiplication: H(jw) = Y(jw) * X_inv(jw)
H_est_k = Y_k .* X_inv_k;

% Inverse FFT to get back to time domain
h_est = ifft(H_est_k, N);

% Make it purely real (removes tiny floating-point imaginary artifacts)
h_est = real(h_est); 


% --- STEP 4: Post-Processing & Truncation ---
% Because X(jw)*X_inv(jw) = e^(-jwT), the direct sound is delayed by T.
% We find that primary impulse peak and extract the true RIR from that point forward.

[~, peak_idx] = max(abs(h_est));

% Define how much of the tail we want to keep (e.g., 1.5 seconds)
tail_length_samples = round(1.5 * Fs); 

% Ensure we don't exceed the array bounds if the recording was cut short
end_idx = min(peak_idx + tail_length_samples, length(h_est)); 

% Truncate to the final Room Impulse Response
h_rir = h_est(peak_idx:end_idx);

% --- STEP 5: Schroeder Backward Integration ---
% Square the impulse response to get power
h_power = h_rir .^ 2;

% Perform backward integration to get the Energy Decay Curve (EDC)
% We reverse the array, compute the cumulative sum, and reverse it back
E = flipud(cumsum(flipud(h_power)));

% --- STEP 6: dB Conversion & Normalization ---
% Normalize energy to its maximum value (which is the first sample)
E_norm = E / max(E);

% Convert to decibel scale. 
% Adding 'eps' (a tiny floating-point number) prevents log10(0) errors at the very end of the tail
EDC_dB = 10 * log10(E_norm + eps);

% --- STEP 7: RT60 Estimation via Linear Regression (T20 Extrapolation) ---
% We evaluate the slope between -5 dB and -25 dB to avoid the initial direct 
% sound variance and the trailing ambient noise floor.

% Find the array indices closest to -5 dB and -25 dB
[~, idx_5dB]  = min(abs(EDC_dB - (-5)));
[~, idx_25dB] = min(abs(EDC_dB - (-25)));

% Define the time vector for the truncated RIR
t_rir = (0:length(h_rir)-1)/Fs;

% Extract the specific time and dB values for this evaluation region
t_linear = t_rir(idx_5dB:idx_25dB);
EDC_linear = EDC_dB(idx_5dB:idx_25dB);

% Perform linear regression (find the best fit line: y = mx + c)
% polyfit(x, y, 1) returns [slope, intercept]
p = polyfit(t_linear, EDC_linear, 1);
slope = p(1);
intercept = p(2);

% Calculate RT60
% Since slope is dB/second, the time for a full 60dB drop is exactly -60 / slope
RT60 = -60 / slope;

fprintf('Estimated RT60: %.3f seconds\n', RT60);

% Plot the result to verify
%figure;
%plot((0:length(h_rir)-1)/Fs, h_rir);
%title('Estimated Room Impulse Response (RIR)');
%xlabel('Time (seconds)');
%ylabel('Amplitude');
%grid on;


figure('Name', 'Signal Pipeline Visualization', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 800]);

% 1. Plot Forward Sweep: x[n]
subplot(4,1,1);
t_x = (0:length(x)-1)/Fs; % Converting the Raw samples into seconds
plot(t_x, x);
title('1. Forward Log Sine Sweep: x[n]');
xlabel('Time (s)'); 
ylabel('Amplitude');
grid on;

% 2. Plot Inverse Filter: x_inv[n]
subplot(4,1,2);
t_xinv = (0:length(x_inv)-1)/Fs; % Converting the Raw samples into seconds
plot(t_xinv, x_inv);
title('2. Amplitude-Modulated Inverse Filter: x_{inv}[n]');
xlabel('Time (s)'); 
ylabel('Amplitude');
grid on;

% 3. Plot Recorded Signal: y[n]
subplot(4,1,3);
t_y = (0:length(y)-1)/Fs; % Converting the Raw samples into seconds
plot(t_y, y);
title('3. Raw Microphone Recording: y[n] (Sweep + Room Reverb)');
xlabel('Time (s)'); 
ylabel('Amplitude');
grid on;

% 4. Plot Raw Estimated RIR: h_est[n]
subplot(4,1,4);
t_hest = (0:length(h_est)-1)/Fs; % Converting the Raw samples into seconds
plot(t_hest, h_est);
title('4. Raw IFFT Output: h_{est}[n]');
xlabel('Time (s)'); 
ylabel('Amplitude');
grid on;

figure('Name', 'Final Result', 'NumberTitle', 'off');
plot(t_rir, h_rir);
title('Final Truncated Room Impulse Response: h_{rir}[n]');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

%5. Final Visualization
figure('Name', 'Schroeder Integration & RT60 Estimation', 'NumberTitle', 'off');

% Plot the actual Energy Decay Curve
plot(t_rir, EDC_dB, 'b', 'LineWidth', 1.5);
hold on;

% Plot the extrapolated linear regression line
line_fit = slope * t_rir + intercept; % y = mx + c
plot(t_rir, line_fit, 'r--', 'LineWidth', 1.5);

% Highlight the specific region used for regression (-5dB to -25dB)
plot(t_linear, EDC_linear, 'g', 'LineWidth', 2);

% Formatting
title(sprintf('Energy Decay Curve | Estimated RT60 = %.3f s', RT60)); %sprintf = String printf
xlabel('Time (seconds)');
ylabel('Energy (dB)');
ylim([-80 5]); % -80 : buttom limit of the graph
% 5 : Top limit of the graoh 
xlim([0 t_rir(end)]); % starts at 0 and ends at the tail's end of time array t_rir
legend('Schroeder Energy Decay Curve', 'Extrapolated Fit Line', 'T20 Evaluation Region');
%legend creates a reference box for identifying each curve with its colour
%and type of line segment
grid on;