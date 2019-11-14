import numpy as np


class logRegression:

	def __init__(self, num_features, alpha, iter, regularized, lam):
		self.w=np.random.rand(num_features+1)
		self.alpha=alpha
		self.i=iter
		self.regularized=regularized
		self.lam=lam

	def logfunc(self, a):

		return (1/(1+np.exp(-1*a)))
		

	def sum(self, x):
		
		sum=0
	
		for i in range(len(x)):
			sum=sum + self.w[i]*x[i]
		return sum


	def loglikelihood(self, x, y):
		
		s=0
		x=np.insert(x, 0, 1, axis=1)

		for i in range(len(y)):

			s=s + y[i]*np.log(self.logfunc(self.sum(x[i]))) + (1-y[i])*np.log(1-self.logfunc(self.sum(x[i])))
		return s
 

	def crossentropyLoss(self, x, y):

		if self.regularized:

			return -1*self.loglikelihood(x, y)
		else:
			extraTerm=0
			for i in range(len(self.w)):
				extraTerm+=np.abs(w[i])

			return -1*self.loglikelihood(x,y) + extraTerm 


	def updateWeights(self, x , y, a):
		
		x=np.insert(x, 0, 1, axis=1)
		vec=np.array([])

		for i in range(len(self.w)):
			vec=np.append(vec,0)


		for i in range(len(y)):

			if self.regularized:
				
				vec=np.add(vec, x[i]*(y[i]-self.logfunc(self.sum(x[i]))))
				vec=np.add(vec, -self.lam*self.w)

			else:
				vec=np.add(vec, x[i]*(y[i]-self.logfunc(self.sum(x[i]))))

		vec=(a)*vec
		
		vec=np.add(self.w, vec)

		self.w=vec

		return None		



	def predict(self,x):
		
		x=np.insert(x,0,1, axis=1)

		y_hat=np.array([])

		for i in range(len(x)):

			res=self.logfunc(self.sum(x[i]))
			if res>=0.5:
				y_hat=np.append(y_hat, 1)
			else:
				y_hat=np.append(y_hat, 0)
		return y_hat


	def fit(self,x,y):

		a=self.alpha
		for i in range(0,self.i):

			self.updateWeights(x, y, a)
			a=(self.alpha)/(i+2)
		return None


	def predict_on(self, k, x, y):

		self.fit(x[np.arange(len(x))!=k], y[np.arange(len(y))!=k])
		
		if self.logfunc(self.sum(x))>=0.5:
			return 1
		else:
			return 0


	def evaluate_acc(self, y, y_hat):

		correct=0

		for i in range(len(y)):
			if y[i]==y_hat[i]:
				correct=correct+1
		
		percent= float(correct)/float(len(y))				

		print(str(percent*100)+'%')



