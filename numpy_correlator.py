import numpy as np
import matplotlib.pyplot as plt
import time

def main():
	benchsizes= [256,512,1024,2048,4096,
						8192,16384,32768,65536,131072,
						262144,524288,1048576,2097152,4194304]
	resFile = open("cpures.csv","w")
	
	for k in range(2,17):
		N_INPUTS = k
		N_OUTPUTS = int(N_INPUTS*(N_INPUTS+1)/2)
		resFile.write(str(k)+"\n")
		for j in range(15):
			SIG_SIZE = benchsizes[j]
			
			#Generating the input signals
			inputArray = [None]*N_INPUTS
			outputArray = [None]*N_OUTPUTS
			for i in range(N_INPUTS):
				inputArray[i] = sigGen(150.0, 8000, SIG_SIZE, True)
				
			ffttime=0;
			multtime=0;
			
			for i in range(11):
				totalstart = time.clock()
				fftstart = time.clock()
				
				RFFT(inputArray)
				
				fftend = time.clock()
				multstart = time.clock()
				
				crossMult(inputArray, outputArray)
				
				multend = time.clock()
				totalend = time.clock()
				
				if(i!=0):
					ffttime=ffttime+(fftend-fftstart)
					multtime=multtime+(multend-multstart)
					
			resFile.write(str(SIG_SIZE)+", "+str(ffttime*1000/10.0)+", "+str(multtime*1000/10.0)+"\n")
			#print("FFT Time: "+str(ffttime*1000/10.0)+" ms")	
			#print("Mult Time: "+str(multtime*1000/10.0)+" ms")
			
		resFile.write("\n")
	return 0
    
    
def sigGen(f, fs, size, noise=False):
	n = np.arange(size)
	if (noise):
		y = np.sin(2*np.pi*f*n/fs) + 0.5*np.random.normal(0, 1, size)
	else:
		y = np.sin(2*np.pi*f*n/fs)
		
	return y

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
