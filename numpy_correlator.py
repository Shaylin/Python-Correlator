import numpy as np
import matplotlib.pyplot as plt
import time

def main():
	N_INPUTS = 2
	N_OUTPUTS = int(N_INPUTS*(N_INPUTS+1)/2)
	SIG_SIZE = 1000
	
	#Generating the input signals
	inputArray = [None]*N_INPUTS
	outputArray = [None]*N_OUTPUTS
	for i in range(N_INPUTS):
		inputArray[i] = sigGen(150.0, 8000, SIG_SIZE, True)

	plt.figure(1)
	for i in range(N_INPUTS):
		plt.subplot(N_INPUTS,1,i+1)
		plt.plot(inputArray[i])
	plt.show()
	
	totalstart = time.clock()
	fftstart = time.clock()
	
	RFFT(inputArray)
	
	fftend = time.clock()
	multstart = time.clock()
	
	crossMult(inputArray, outputArray)
	
	multend = time.clock()
	totalend = time.clock()
	
	print("FFT Time: "+str((fftend-fftstart)*1000)+" ms")	
	print("Mult Time: "+str((multend-multstart)*1000)+" ms")
	print("Total Time: "+str((totalend-totalstart)*1000)+" ms")
	
	#plt.figure(2)
	#for i in range(N_INPUTS):
		#plt.subplot(N_INPUTS,1,i+1)
		#plt.plot(np.abs(inputArray[i]))
	#plt.show()
	
	#plt.figure(3)
	#for i in range(N_OUTPUTS):
		#plt.subplot(N_OUTPUTS,1,i+1)
		#plt.plot(np.abs(outputArray[i]))
	#plt.show()
	
	return 0
    
    
    
def sigGen(f, fs, size, noise=False):
	n = np.arange(size)
	if (noise):
		y = np.sin(2*np.pi*f*n/fs) + 0.5*np.random.normal(0, 1, size)
	else:
		y = np.sin(2*np.pi*f*n/fs)
		
	padAmount = int(2**np.ceil(np.log2(2*size)))
	yzer = np.zeros(padAmount)
	
	for i in range(size):
		yzer[i] = yzer[i]+y[i]
	
	return yzer

def RFFT(inputArray):
	for i in range(len(inputArray)):
		inputArray[i] = np.fft.rfft(inputArray[i])

def crossMult(inputArray, outputArray):
	count = 0
	for i in range(len(inputArray)):
		for j in range(i, len(inputArray)):
			outputArray[count] = inputArray[i]*np.conj(inputArray[j])
			count = count + 1

if __name__ == "__main__":
    main()
