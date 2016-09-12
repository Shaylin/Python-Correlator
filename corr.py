import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import time

def main():
	FREQ = 10.0
	SAMPLING_FREQ = 1000.0
	TIME = 0.5
	DELAY =0
	
	
	#Verify Signal Generation (it does make the length the next power of 2)
	t,sig = sigGen(FREQ, SAMPLING_FREQ, TIME, DELAY)
	plt.figure(1)
	plt.title("Generated 10Hz Sine and Cosine")
	plt.plot(t, np.real(sig),'b', label="Cosine")
	plt.plot(t, np.imag(sig), 'r', label="Sine")
	plt.ylabel("Amplitude [V]")
	plt.xlabel("Time [s]")
	plt.axis("tight")
	plt.legend(loc=0,frameon=False)
	plt.grid()
	
	#Verify Fourier Transform
	[xPol, yPol] = [np.real(sig), np.imag(sig)]
	w,X = fourier(xPol, SAMPLING_FREQ, None)
	w,Y = fourier(yPol, SAMPLING_FREQ, None)
	yAmp = np.abs(X)
	xAmp = np.abs(Y)
	plt.figure(2)
	lab = 'Peak amplitude X (Cosine) %.3f at %.2f [Hz]' % (np.max(xAmp), w[np.argmax(xAmp)])
	plt.plot(w, xAmp, 'b', label=lab)
	lab = 'Peak amplitude Y (Sine) %.3f at %.2f [Hz]' % (np.max(yAmp), w[np.argmax(xAmp)])
	plt.plot(w, yAmp, 'r--', label=lab)
	plt.ylabel("Amplitude [V]")
	plt.xlabel("Frequency [Hz bins]")
	plt.legend(loc=0,frameon=False)
	plt.axis("tight")
	
	#Verify Cross Power Spectrum
	plt.figure(3)
	plt.subplot(2,1,1)
	XX = correlate(X,X) 
	XY = correlate(X,Y)
	YY = correlate(Y,Y)
	
	lab = 'Peak amplitude XX %.3f at %.2f [Hz]' % (np.max(np.abs(XX)), w[np.argmax(np.abs(XX))])
	plt.plot(w, np.abs(XX), 'b', label=lab)
	lab = 'Peak amplitude XY %.3f at %.2f [Hz]' % (np.max(np.abs(XY)), w[np.argmax(np.abs(XY))])
	plt.plot(w, np.abs(XY), 'k--', label=lab)
	lab = 'Peak amplitude YY %.3f at %.2f [Hz]' % (np.max(np.abs(YY)), w[np.argmax(np.abs(YY))])
	plt.plot(w, np.abs(YY), 'r-', label=lab)
	plt.ylabel("Amplitude [V]")
	plt.legend(loc=0,frameon=False)
	
	plt.subplot(2,1,2)
	plt.title('Time delay between X and Y cause a phase slope')
	angle = np.unwrap(np.angle(XX))
	lab = r'$\angle(XX)$'
	plt.plot(w, np.rad2deg(angle), 'b:', label=lab)
	lab = r'$\angle(XY)$'
	angle = np.unwrap(np.angle(XY))
	plt.plot(w, np.rad2deg(angle), 'k--', label=lab)
	lab = r'$\angle(YY)$'
	angle = np.unwrap(np.angle(YY))
	plt.plot(w, np.rad2deg(angle), 'r-', label=lab)
	plt.legend(loc=0,frameon=False)
	plt.ylabel('Phase')
	plt.xlabel("Frequency [Hz bins]")
	
	#Verify Time Domain Correlation
	
	#Non-padded version
	xx = (np.fft.fftshift(np.fft.irfft(XX)))
	xy = (np.fft.fftshift(np.fft.irfft(XY)))
	yy = (np.fft.fftshift(np.fft.irfft(YY)))
	corrtime = np.arange(len(xx)/-2.0,len(xx)/2.0)/SAMPLING_FREQ
	
	plt.figure(4)
	plt.subplot(3,1,1)
	plt.plot(corrtime, xx, 'b', label="xx")
	plt.plot(corrtime, xy, 'k', label="xy")
	plt.plot(corrtime, yy, 'r', label="yy")
	plt.axvline(x = (np.argmax(np.abs((xx)))-len(xx)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.axvline(x = (np.argmax(np.abs((xy)))-len(xy)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.xlabel("Time [s]")
	plt.title("Without zero padding")
	plt.ylabel("Magnitude [V]")
	plt.legend(loc=0,frameon=False)
	
	plt.subplot(3,1,2)
	plt.plot(corrtime, np.abs(xx), 'b', label="abs(xx)")
	plt.plot(corrtime, np.abs(xy), 'k', label="abs(xy)")
	plt.plot(corrtime, np.abs(yy), 'r', label="abs(yy)")
	plt.axvline(x = (np.argmax(np.abs((xx)))-len(xx)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.axvline(x = (np.argmax(np.abs((xy)))-len(xy)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.xlabel("Time [s]")
	plt.ylabel("Magnitude [V]")
	plt.legend(loc=0,frameon=False)
	
	plt.subplot(3,1,3)
	offset = np.abs(np.argmax(np.abs((xx))) - np.argmax(np.abs((xy))))
	time_resolution = np.mean(np.diff(t))
	delay = offset*time_resolution
	plt.plot(t, np.real(sig),'b', label="Cosine")
	plt.plot(t, np.imag(sig), 'r', label="Sine")
	plt.ylabel("Amplitude [V]")
	plt.xlabel("Time [s]")
	plt.title('y signal has %.0f ms delay' % (delay*1e3))
	plt.axvline(x=delay, color='k', linestyle='--')
	plt.legend(loc=0,frameon=False)
	plt.tight_layout()
	
	#Padded version
	padW,padX = fourier(xPol, SAMPLING_FREQ, 2*len(xPol))
	padW,padY = fourier(yPol, SAMPLING_FREQ, 2*len(yPol))
	padXX = correlate(padX, padX)
	padXY = correlate(padX, padY)
	padYY = correlate(padY, padY)
	xx = (np.fft.fftshift(np.fft.irfft(padXX)))
	xy = (np.fft.fftshift(np.fft.irfft(padXY)))
	yy = (np.fft.fftshift(np.fft.irfft(padYY)))
	corrtime = np.arange(len(xx)/-2.0,len(xx)/2.0)/SAMPLING_FREQ
	
	plt.figure(5)
	plt.subplot(3,1,1)
	plt.title("With zero padding")
	plt.plot(corrtime, xx, 'b', label="xx")
	plt.plot(corrtime, xy, 'k', label="xy")
	plt.plot(corrtime, yy, 'r', label="yy")
	plt.axvline(x = (np.argmax(np.abs((xx)))-len(xx)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.axvline(x = (np.argmax(np.abs((xy)))-len(xy)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.xlabel("Time [s]")
	plt.ylabel("Magnitude [V]")
	plt.legend(loc=0,frameon=False)
	
	plt.subplot(3,1,2)
	plt.plot(corrtime, np.abs(xx), 'b', label="abs(xx)")
	plt.plot(corrtime, np.abs(xy), 'k', label="abs(xy)")
	plt.plot(corrtime, np.abs(yy), 'r', label="abs(yy)")
	plt.axvline(x = (np.argmax(np.abs((xx)))-len(xx)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.axvline(x = (np.argmax(np.abs((xy)))-len(xy)/2)/SAMPLING_FREQ, color='k', linestyle='--')
	plt.xlabel("Time [s]")
	plt.ylabel("Magnitude [V]")
	plt.legend(loc=0,frameon=False)
	
	plt.subplot(3,1,3)
	offset = np.abs(np.argmax(np.abs((xx))) - np.argmax(np.abs((xy))))
	time_resolution = np.mean(np.diff(t))
	delay = offset*time_resolution
	plt.plot(t, np.real(sig),'b', label="Cosine")
	plt.plot(t, np.imag(sig), 'r', label="Sine")
	plt.ylabel("Amplitude [V]")
	plt.xlabel("Time [s]")
	plt.title('y signal has %.0f ms delay' % (delay*1e3))
	plt.axvline(x=delay, color='k', linestyle='--')
	plt.legend(loc=0,frameon=False)
	plt.tight_layout()
	
	plt.show()
	


def sigGen(f, fs, t, td=0):
	"""
	Generate both a sine and cosine of amplitude 1 for simulation purposes
	
	Parameters:
	f -- [Hz] The signal frequency
	fs -- [Hz] The sampling frequency
	t -- [s] The time length of the signal 
	td -- [s] The time delay of the signal
	"""
	N = int(fs*t)
	size = int(2**np.ceil(np.log2(N)))
	sampletime = np.arange(0,size)/fs
	sinewave = np.sin(2.*np.pi*f*sampletime)
	cosinewave = np.cos(2.*np.pi*f*sampletime)
	
	return [sampletime, np.vectorize(complex)(cosinewave, sinewave)]

	
def fourier(data, fs, n=None):
	"""
	Generate the one-sided discrete fourier transform of a real signal
	
	Parameters:
	data -- The input signal
	fs -- [Hz] The sampling frequency
	n -- Number of channels of the fourier transform
	"""
	L = len(data)
	if n is None:
		F = np.fft.rfft(data)*2/L
		freqs = np.fft.rfftfreq(L, 1.0/fs)
	else:
		F = np.fft.rfft(data, n)*2/L
		freqs = np.fft.rfftfreq(n, 1.0/fs)
	
	if not np.iscomplexobj(F):
		 F = np.array(F, dtype=np.complex)
	
	return [freqs, F]
	
def correlate(X, Y):
	"""
	Returns the multiplication of the first signal and complex conjugate of the
	second signal
	
	Parameters:
	X -- First input signal
	Y -- Second input signal
	"""
	return X*np.conj(Y)

if __name__ == "__main__":
    main()
