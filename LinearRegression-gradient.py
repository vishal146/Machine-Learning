import numpy as np
import matplotlib.pyplot as plt

def plot(x,y,b,m):

	#plot all the data points in the scatter plot
	plt.scatter(x,y,color="m", marker="o", s=20)
	
	#Now calculate values of our predicted line
	y_pred = m*x + b
	#Now plot the line
	plt.plot(x,y_pred, color='g')

	plt.xlabel('x')
	plt.ylabel('y')
	plt.show(block=False)
	plt.pause(0.2)

	#uncomment to clear the graph each time
	#plt.close()

def gradient(x, y, b, m, learning_rate):
	b_gradient = 0
	m_gradient = 0
	for i in range(len(x)):
		#summation of dL/db, differentiating loss function w.r.t. constant(b)
		b_gradient += (2*(m*x[i] + b - y[i]))
		#summation of dL/dm, , differentiating loss function w.r.t. slope(m)
		m_gradient += (2*x[i]*(m*x[i] + b - y[i]))
	new_b = b - (learning_rate*b_gradient)
	new_m = m - (learning_rate*m_gradient)
	return [new_b, new_m]

def calculateNewCoeff(x, y, b, m, num_iteration, learning_rate):
	for i in range(num_iteration):
		[b,m] = gradient(x,y,b,m,learning_rate)
		plot(x,y,b,m)
		#print('m: ', m, '  b: ', b)
	return [b,m]


def main():
    learning_rate = 0.0008
    #sample data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    intial_b = 0
    intial_m = -1
    num_iteration = 30

    plot(x,y,intial_b,intial_m)
    [b,m] = calculateNewCoeff(x,y, intial_b, intial_m, num_iteration, learning_rate)
    plot(x,y,b,m)

if __name__=="__main__":
	main()