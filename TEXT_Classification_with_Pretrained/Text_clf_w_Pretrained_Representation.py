#codeช้ Deep Learning (DL) ในขั้นตอนการสกัด features ก่อนที่จะนำ features เหล่านั้นไปเทรนโมเดล Machine Learning (ML)

import sklearn.experimental
import sklearn.externals
import sklearn.externals._arff
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
import tensorflow as tf 
import tensorflow_hub  as hub
import tensorflow_text
import tqdm
import pandas as pd
import sklearn
import numpy as np
import joblib
#Universal Sentence Encoder Multilingual (USE Multilingual) ถูกสร้างขึ้นโดยใช้เทคนิค Deep Learning หลายอย่างร่วมกัน 
embeding_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')   #Universal Sentence Encoder (USE) เป็น embedding model หนึ่ง โดยหน้าที่หลักของมันคือการแปลงข้อความ (เช่น ประโยคหรือย่อหน้า) ให้อยู่ในรูปแบบเวกเตอร์ที่มีมิติสูง (512 มิติ) 
                            # ซึ่งเวกเตอร์เหล่านี้สามารถใช้ในการทำงานต่าง ๆ ของการประมวลผลภาษาธรรมชาติ (NLP) เช่น การจำแนกประเภทข้อความ การคำนวณความเหมือนเชิงความหมาย การจัดกลุ่ม และงานอื่น ๆ

# setup parameter

train_df = pd.read_csv("train_df.csv",encoding="utf-8")
test_df = pd .read_csv("test_df.csv",encoding="utf-8")

train_set_df,val_set_df = sklearn.model_selection.train_test_split(train_df,test_size=0.15,shuffle=True,random_state=1412)

train_set_df = train_set_df.reset_index(drop=True)
val_set_df = val_set_df.reset_index(drop=True)

y_train = train_set_df.categories
y_val= val_set_df.categories

x_train = []
x_val= [] 
batch_size = 10
# encode and Embbeding

for i in tqdm.tqdm(range(y_train.shape[0]//batch_size+1)): #ทำการ Embbeding ข้อมูล ที่ลำ batch
    x_train.append(embeding_model(train_set_df.texts[(i*batch_size):((i+1)*batch_size)]).numpy())

for i in tqdm.tqdm(range((val_set_df.shape[0]//batch_size+1))):
    x_val.append(embeding_model(val_set_df.texts[(i*batch_size):((i+1)*batch_size)]).numpy())


x_train = np.concatenate(x_train,axis=0)
x_val = np.concatenate(x_val,axis=0)

model = sklearn.svm.LinearSVC(penalty="l2",class_weight="balanced",C=2)
model.fit(X=x_train,y=y_train)

# Save model

model_filename = 'linear_svc_model.pkl'

joblib.dump(model, model_filename)

y_val_pred = model.predict(x_val)
#print(y_val_pred)
print(sklearn.metrics.classification_report(y_pred=y_val_pred,y_true=y_val))


###################################################################### Test Text_clf_w_Pretrained_Representation.py  ##########################################################

loaded_model = joblib.load(r'D:\machine_learning_AI_Builders\บท4\NLP\linear_svc_model.pkl')
text = "ร้านอาหารร้านนี้อร่อยมาก"
embedded_text = np.array(embeding_model([text]))  # Embed the text using USE
prediction = loaded_model.predict(embedded_text)  # Make a prediction

print(f"Predicted category: {prediction[0]}")

