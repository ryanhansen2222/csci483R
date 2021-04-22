import random
import numpy as np
import copy


class DataGen():
	
	'''
	For some matrix A, generates synthetic data from randomly generated
	data
	''' 


	def __init__(self, equations):
	
		self.equations = equations
		changingeqtns = copy.deepcopy(self.equations)#Dont modify originals
		self.permutedequations = self.addcolumn(changingeqtns)


   
	#Add gaussian noise to measurement, scaled by its own magnitude
	def noise(self, rawmeasurements):
			noisy = []
			for  measurement in rawmeasurements:
				noisy.append(measurement[0,0]*(1+.05*random.gauss(0, 1)))
			return noisy 

	#EQTN
	#original : [A]x = b
	#picks values for x 
	def generatex(self, eqtns):
		length = len(eqtns[0])
		x = [random.uniform(1,10) for a in range(length)]
		return x
		
		
	   
	#EQTN
	#original : [A]x = b
	#and solves for b
	def findrawmeas(self, listx, eqtns):
			A = np.matrix(eqtns)
			x = np.matrix(listx)
			rawmeasvals = A.dot(np.transpose(x)) #(b)
			return rawmeasvals
		 
 
	  
	#Generate full set of synthetic data
	#For a given generated x value
	def synthesizereal(self):

			dataset = []

			#Make x, make b, then add noise to b {{numdatapoints}} times
			x = self.generatex(self.equations)
			rawmeas = self.findrawmeas(x,self.equations)

			datapoint = self.noise(rawmeas)


			return datapoint, x 

	def synthesizefake(self ):


			#Make x, make b, then add noise to b {{numdatapoints}} times
			x = self.generatex(self.permutedequations)
			rawmeas = self.findrawmeas(x,self.permutedequations)

			datapoint = self.noise(rawmeas)


			return datapoint, x 


			
	def addcolumn(self, eqtns):


		#make equations coeff account for new column
		for e in eqtns:
			pos = random.choice([-1,1])
			coeff = random.randint(1,10)
			e.append(pos*coeff)
		return eqtns




if __name__ == "__main__":
	eqtns = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]]
	test = DataGen(eqtns)
	
	dataset, x = test.synthesizereal()

	
	baddata, badx = test.synthesizefake()

	print('From equations: ', test.equations, 'And x:', x)
	print('Good Data: ', dataset)

	print('From equations: ', test.permutedequations, 'And x:', badx)
	print('Bad data: ', baddata)
