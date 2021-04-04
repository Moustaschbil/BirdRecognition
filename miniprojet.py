import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from PIL import Image 
import numpy as np

#variables globales
tab_class = ["ALEXANDRINE PARAKEET", "ANTBIRD", "AFRICAN FIREFINCH", "EASTERN MEADOWLARK", "HOUSE FINCH", "SNOWY OWL"]
nb_class = len(tab_class)
x_train = np.empty((nb_class*150, 224, 224, 3))
y_train = np.empty((nb_class*150))

#fonctions
def prep_image(filepath):
	imgpil = Image.open(filepath) 
	imgpil_rsz = imgpil.resize((224, 224)) 
	img = np.array(imgpil_rsz, dtype=float)
	imgbgr = img[...,::-1].copy()
	imgbgr[:, :, 0] -= 103.939
	imgbgr[:, :, 1] -= 116.779
	imgbgr[:, :, 2] -= 123.68
	return imgbgr

def prep_class():
	global x_train, y_train
	for class_nb in range(nb_class):
		classname = tab_class[class_nb]
		for x in range(150):
			fp = r'./{}/{:0>3}.jpg'.format(classname, x+1)
			x_train[nb_class*x + class_nb] = prep_image(fp)
			y_train[nb_class*x + class_nb] = class_nb

#preparation des images
prep_class()
y_train = keras.utils.to_categorical(y_train, nb_class)

#creation du reseau
vgg16_features = keras.applications.vgg16.VGG16(include_top= False, weights= 'imagenet')
vgg16_features.trainable = False
inputs = Input(shape=(224, 224, 3))
out_vggfeatures = vgg16_features(inputs)
out_flat = Flatten()(out_vggfeatures)
out_hidden1 = Dense(512, activation='relu')(out_flat)
drop1 = Dropout(0.2)(out_hidden1)
out_hidden2 = Dense(64, activation='relu')(out_flat)
drop2 = Dropout(0.2)(out_hidden2)
prediction = Dense(nb_class, activation='softmax')(out_hidden2)
model = Model(inputs= inputs, outputs= prediction)
model.summary()
model.compile(loss= 'categorical_crossentropy',
	 		  optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

#entrainement du reseau
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

import matplotlib.pyplot as plt
xvals = range(10)
plt.clf()
plt.plot(xvals, history.history['acc'], label='Training accuracy')
plt.plot(xvals, history.history['val_acc'], label='Validation accuracy')
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
