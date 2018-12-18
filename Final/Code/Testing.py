from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score
model = load_model('D:\Result\DAI\lungdetection-rate-0001.h5')

gen = ImageDataGenerator()

test_batches = gen.flow_from_directory("../Data/Test", model.input_shape[1:3], shuffle=False,
                                       color_mode="grayscale", batch_size=8)

p = model.predict_generator(test_batches, verbose=True)


pre = pd.DataFrame(p)
pre["filename"] = test_batches.filenames
pre["label"] = (pre["filename"].str.contains("Training")).apply(int)
pre['pre'] = (pre[1]>0.5).apply(int)

recall_score(pre["label"],pre["pre"])

roc_auc_score(pre["label"],pre[1])

tpr,fpr,thres = roc_curve(pre["label"],pre[1])
roc = pd.DataFrame([tpr,fpr]).T
roc.plot(x=0,y=1)