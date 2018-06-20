import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import optimizers
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten
import subprocess
import matplotlib.pyplot
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Model
from sklearn import svm
from sklearn.metrics import accuracy_score
global svm_x_train 
global svm_y_train 
global model
global model2
global dummy2_y
import pickle
def model():
	model = Sequential()
	model.add(Conv2D(8, (3, 3), padding='same',
	                 input_shape=(513, 800, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(8, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(16, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(16, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	#model.add(Dense(10))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(57))
	model.add(Activation('softmax'))

	model.summary()

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	model.load_weights('my_model_weights.h5')

	return model
# rootdir = path to training data dir
def training_data(rootdir):
	spectograms = []
	spect_read = []
	spectograms_ids = []
	for subdir, dirs, files in os.walk(rootdir):
	    for file in files:
	        if file.endswith('png'): 
	            try:
	                x = plt.imread(subdir+'/'+file)
	            except:
	                continue 
	            if str(x.shape) == '(513, 800, 3)': 
	                spect_read.append(x)
	                #print(subdir) 
	                name = subdir.replace(rootdir, '')
	                #print(name)
	                #name = name.replace('/spects', "")
	                spectograms_ids.append(name)
	                spectograms.append(file)
	temp = []
	temp.append(spect_read)
	temp.append(spectograms_ids)
	return temp
	# print(y_train)
# rootdir1 = path to test data dir
def test_data(rootdir1):
	spectograms = []
	spect_read = []
	spectograms_ids = []
	for subdir, dirs, files in os.walk(rootdir1):
	    for file in files:
	        if file.endswith('png'): 
	            try:
	                x = plt.imread(subdir+'/'+file)
	            except:
	                continue
	            if str(x.shape) == '(513, 800, 3)': 
	                spect_read.append(x)
	                name = subdir.replace(rootdir1, '')
	                #name = name.replace('/spects', "")
	                spectograms_ids.append(name)
	                spectograms.append(file)
	temp = []
	temp.append(spect_read)
	temp.append(spectograms_ids)
	return temp
	# print(y_test)
def fit_data(y_train,y_test,x_train, model):
	encoder = LabelEncoder()
	y_temp_train = y_train
	encoder.fit(y_temp_train)
	encoded_Y = encoder.transform(y_temp_train)
	dummy_y = np_utils.to_categorical(encoded_Y)
	svm_x_train = []
	svm_y_train = []
	y_temp2_train = y_test
	encoder.fit(y_temp2_train)
	encoded_Y = encoder.transform(y_temp2_train)
	dummy2_y = np_utils.to_categorical(encoded_Y)
	model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
	for i in range(len(x_train)):
	    x_1 = np.expand_dims(x_train[i], axis=0)
	    flatten_2_features = model2.predict(x_1)
	    svm_x_train.append(flatten_2_features)
	    svm_y_train.append(dummy_y[i])

	temp = []
	temp.append(svm_x_train)
	temp.append(svm_y_train)
	temp.append(dummy2_y)
	return temp

def svm_train(svm_x_train,svm_y_train):
	clf.fit(svm_x_train, svm_y_train)
	pickle.dump(clf,open('clf.p','wb'))


def extractData(svm_x_train,svm_y_train):
	svm_x_train = np.array(svm_x_train)
	clf = svm.SVC(kernel='rbf', class_weight='balanced')
	dataset_size = len(svm_x_train)
	svm_x_train = np.array(svm_x_train).reshape(dataset_size,-1)
	svm_y_train = np.array(svm_y_train)
	svm_y_train = [np.where(r==1)[0][0] for r in svm_y_train]
	return svm_x_train,svm_y_train


def svm_test(svm_x_train,svm_y_train,x_test,model, dummy2_y):
	clf = pickle.load(open('clf.p','rb'))
	svm_y_test = []
	svm_x_test = []
	model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
	for i in range(len(x_test)):
	    x_1 = np.expand_dims(x_test[i], axis=0)
	    #x_1 = preprocess_input(x_1)
	    flatten_2_features = model2.predict(x_1)
	    svm_x_test.append(flatten_2_features)
	    svm_y_test.append(dummy2_y[i])
	svm_x_test = np.array(svm_x_test)
	dataset_size = len(svm_x_test)
	svm_x_test = np.array(svm_x_test).reshape(dataset_size,-1)
	svm_y_test = [np.where(r==1)[0][0] for r in svm_y_test]
	predicted  = clf.predict(svm_x_train)
	print(predicted)
	print(accuracy_score(svm_y_train, predicted))



def main(testbool):
	mod = model()
	rootdir = 'data/train/'
	traindata = training_data(rootdir)
	x_train = traindata[0]
	y_train = traindata[1]

	rootdir1 = 'data/test/'
	testdata = test_data(rootdir1)
	x_test = testdata[0]
	y_test = testdata[1]

	svm = fit_data(y_train,y_test,x_train,mod)
	svm_x_train = svm[0]
	svm_y_train = svm[1]
	dum = svm[2]
	svm_x_train,svm_y_train = extractData(svm_x_train,svm_y_train)
	if testbool:
		svm_test(svm_x_train,svm_y_train,x_test,mod, dum)
	else:	
		svm_train(svm_x_train,svm_y_train)
	
if __name__ == '__main__':
	main()