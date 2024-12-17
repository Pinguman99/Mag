import keras.utils.np_utils

print("test")

import glob
import numpy as np
import cv2

train_images=[]
train_labels=[]

for filename in glob.glob("images\\X\\*.png"):
    im = cv2.imread(filename)
    train_images.append(np.asarray(im))
    train_labels.append(np.array((1, 0)))

for filename in glob.glob("images\\O\\*.png"):
    im = cv2.imread(filename)
    train_images.append(np.asarray(im))
    train_labels.append(np.array((0, 1)))


import model

net=model.make_net()

net.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_images=np.asarray(train_images)
train_labels=np.asarray(train_labels)


nchannels=3
nclasses=2
sz=30

nsamples=train_images.shape[0]


#TODO: отладить ошибку
# for epoch in range(10):
#     for ind in range(nsamples):
#         img = train_images[ind:ind+1]
#         lbl = train_labels[ind:ind+1]
#         img = np.reshape(img, (1, sz, sz, nchannels))
#         lbl = np.reshape(lbl, (1, nclasses))
#         net.fit(img, lbl)
#
#
# net.save_weights("net.h5")

net=model.make_net()
# net.load_weights("net.h5")


for i in range(nsamples):
    print(f"Sample {i}:")
    input=train_images[i:i+1]
    predictions = net.predict(input)
    print(predictions)
    etalon=train_labels[i]
    print(etalon)
