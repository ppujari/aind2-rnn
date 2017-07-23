import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    size = series.shape[0]
    start = 0
    while (start < size - window_size):
        a = []
        b = []
        count = 0 
        while ( count < window_size ):
            a.append(series[start + count])
            count = count + 1
        b.append(series[start + count])
        start = start + 1
        X.append(a)
        y.append(b)
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
### TODO: create required RNN model
# import keras network libraries
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
	np.random.seed(0)

# TODO: build an RNN to perform regression on our time series input/output data
	model = Sequential()
	model.add(LSTM(5, input_shape=(window_size, 1), activation='relu'))
	model.add(Dense(1, activation='tanh'))

# build model using keras documentation recommended optimizer initialization
	optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
	model.compile(loss='mean_squared_error', optimizer=optimizer)

### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text


    # remove as many non-english characters and character sequences as you can 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs

    # containers for input/output pairs
    inputs = []
    outputs = []
    
    size = len(text)
    start = 0
    while (start < size - window_size - step_size):
        a = []
        b = []
        count = 0 
        while ( count < window_size ):
            a.append(text[start + count])
            count = count + 1
        step_count = 0
        while ( step_count < step_size):
            b.append(text[start + count + step_count])
            step_count = step_count + 1
        start = start + 1
        inputs.append(a)
        outputs.append(b)
#        print (a)
#        print (b)
    # reshape each 
#    inputs = np.asarray(inputs)
#    inputs.shape = (np.shape(inputs)[0:2])
#    outputs = np.asarray(outputs)
#    outputs.shape = (len(outputs),1)
    inputs = " ".join(str(x) for x in inputs)
    outputs = " ".join(str(x) for x in outputs)

    
    return inputs,outputs
