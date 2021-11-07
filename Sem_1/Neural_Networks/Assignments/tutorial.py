import numpy as np # import library for processing arrays, use short-hand notation "np" for the library
# if you are getting an error here you should: pip install numpy, telling python to install numpy package

# this is a function definition
def add_things_and_print(arg1, arg2):
    print("sum of ", arg1, " and ", arg2, " is ", arg1+arg2)

# this is another function definition, the actual execution starts after this
def my_activation(local_field): # step activation
    return_this = np.zeros(local_field.shape[0]) # create an all zero array with the same dim as the input vec
    dum = np.where(local_field >= 0)[0] # where returns two arrays, one in [0], one in [1] position. Indices are at [0]/
    return_this[dum] = 1 # set positive components to 1 (rest already remain zero)
    return return_this


# the program will "start" here (after importing the libraries and reading your func definitions)
add_things_and_print(3, 4)

a = np.array([2,3,4])  # a three-dimensional vector
print(a)
print(a.shape) # print number of dimensions

b = np.array([[2,3,4]]) # a 1x3 matrix
print(b)
print(b.shape)

c = np.array([[0],[1],[2]]) # a 3x1 matrix
print(c)
print(c.shape)

print(b.dot(c)) # matrix multiplications (note we do not use star *)
print(c.dot(b))

# some neural networking
input_dim = 10
no_neurons_in_hidden_layer = 100
output_dim = 10

# initialize some weight matrices randomly
w1 = np.random.randn(no_neurons_in_hidden_layer, input_dim)
w2 = np.random.randn(output_dim, no_neurons_in_hidden_layer)

# initialize an input randomly
x = np.random.randn(input_dim, 1)

# calculate the output of a two-layer neural net with step activation function.
outputs_after_layer1 = my_activation(np.dot(w1, x))
overall_output = my_activation(np.dot(w2, outputs_after_layer1))
