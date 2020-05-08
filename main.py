import os
from model import *
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from keras.callbacks import ModelCheckpoint
from keras import backend as K



image_npy = r'Training data path'
label_npy = r'label data path'
test_npy = r'test data path'

feature = np.load(image_npy)
label = np.load(label_npy)
test = np.load(test_npy)


kf_number = 0
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(feature):
    kf_number += 1
    x_train, y_train = feature[train_index], label[train_index]
    x_test, y_test = feature[test_index], label[test_index]

    model = FCANet(pretrained_weights=None,img_input = (256,256,1))

    print('training data：',x_train.shape[0])
    print('validation data：',x_test.shape[0])
    print('testing data：',test.shape[0])

    checkpoint = ModelCheckpoint(filepath=r"Best model save path"%kf_number,#(就是你准备存放最好模型的地方),
                                 monitor='val_loss',#(或者换成你想监视的值,比如acc,loss, val_loss,其他值应该也可以,还没有试),
                                 verbose=1,#(如果你喜欢进度条,那就选1,如果喜欢清爽的就选0,verbose=冗余的),
                                 save_best_only='True',#(只保存最好的模型,也可以都保存),
                                 save_weights_only='True',
                                 mode='min',#(如果监视器monitor选val_acc, mode就选'max',如果monitor选acc,mode也可以选'max',如果monitor选loss,mode就选'min'),一般情况下选'auto',
                            period=1)#(checkpoints之间间隔的epoch数)
    callbacks_list = [checkpoint]
    model.fit(x = x_train,y = y_train, batch_size=8, epochs=150, verbose=2,callbacks=callbacks_list,validation_data=(x_test,y_test))
    K.clear_session()



