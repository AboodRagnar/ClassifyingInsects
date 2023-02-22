
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import Sequential,save_model,load_model
from keras.applications.vgg19 import VGG19 
from keras.preprocessing.image import ImageDataGenerator 
import h5py
import matplotlib.pyplot as plt 
import numpy as np
import json
GetImages=ImageDataGenerator()


TrainImges=GetImages.flow_from_directory('D:/AI/Keras/DataSet/BeesVsInsects/kaggle_bee_vs_wasp/Train',target_size=(224,224),classes=['Bee','Wasp','Otherinsect'],batch_size=20)
TestImages=GetImages.flow_from_directory('D:/AI/Keras/DataSet/BeesVsInsects/kaggle_bee_vs_wasp/Test',target_size=(224,224),classes=['Bee','Wasp','Otherinsect'],batch_size=20)
PredictImages=GetImages.flow_from_directory('D:/AI/Keras/DataSet/BeesVsInsects/kaggle_bee_vs_wasp/Predict',target_size=(224,224),classes=['Bee','Wasp','Otherinsect'],batch_size=10)

# Model=Sequential()

# for ly in VGG19().layers[:-1]:
   
#     Model.add(ly)

# for ly in Model.layers:
#      ly.trainable=False
    

# Model.add(Dense(3,activation='softmax'))

# Model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
# h=Model.fit_generator(TrainImges,steps_per_epoch=100,epochs=25,verbose=1,validation_data=TestImages,shuffle=True)
# Model.save('Insect_Bee_Wasp.h5')
def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config']['layers'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

fix_layer0('Insect_Bee_Wasp.h5', [None, 224, 224, 3], 'float32')

Model = load_model('Insect_Bee_Wasp.h5')

LoadImg,l=next(PredictImages)
print(Model.predict_classes(LoadImg,batch_size=5),PredictImages.class_indices)

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

plots(LoadImg)
plt.plot(h.history['accuracy'])
plt.show()