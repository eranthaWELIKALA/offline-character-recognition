import numpy as np
'''lst = [1,2,3]
arr = np.array(lst)
a1 = arr.reshape(3,1)
print(a1)
a2 = arr.T
print(a2)
print(np.dot(a2,a1))
a = np.array([[1.0,9.2],[5.2,2.3]])
b = np.array([[1.0],[1]])
print(a+b)
print(np.zeros((2,2)))
'''

'''
Y_train = np.array([1,2,0,1,0])
array_size = 5
prepared_Y = np.zeros((array_size,3)).reshape(array_size,3)
example_no = 0
for i in Y_train:
	print(i,example_no)
	prepared_Y[example_no,i] = 1
	example_no += 1
print(prepared_Y)
'''

assert (type(1) == float)