import sklearn
import os
import pandas as pd
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import process_text
import joblib

train_data_df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\NLP\train_df.csv") 
test_data_df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\NLP\test_df.csv")

#print(test_data_df.columns)

train_set_df,val_set_df = sklearn.model_selection.train_test_split(train_data_df,test_size=0.15,random_state=1412)
train_set_df = train_set_df.reset_index(drop=True) # drop=True ในฟังก์ชัน reset_index() ของ Pandas หมายถึงการลบคอลัมน์ที่เป็น index เดิมของ DataFrame ออกไป และไม่จัดเก็บคอลัมน์นั้นเป็นคอลัมน์ใหม่ที่มีชื่อ "index" ใน DataFrame ที่ผลลัพธ์ 
val_set_df = val_set_df.reset_index(drop=True)
#print(train_set_df.columns)
#print(val_set_df.columns)

#################################################################### สร้างตาราง BoW  (การสร้าง feature โดยการ Bag of word เพื่อนำไป train ) ################################################################################

#TfidfVectorizer เป็นคลาสในไลบรารี scikit-learn ที่ใช้ในการแปลงข้อความเป็นเวกเตอร์ TF-IDF (Term Frequency-Inverse Document Frequency) ซึ่งเป็นเทคนิคที่ใช้ในการประมวลผลภาษาธรรมชาติ (Natural Language Processing, NLP) 
# โดยสร้างเวกเตอร์แบบ sparse matrix ที่มีการนับความถี่ของคำและน้ำหนัก TF-IDF ของแต่ละคำในเอกสารที่ให้มา เพื่อนำไปใช้ในการวิเคราะห์หรือการเรียนรู้ข้อมูลต่อไปได้ง่ายขึ้น
tfidf = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=process_text.process_text,ngram_range=(1,2),min_df=20,sublinear_tf=True)# tokenizer = ขันตอนการpreprocessed text
                                                                                                                                          # ngram_range = การรวมคำ 1-2 คำ
                                                                                                                                          # min_df = คำนั้นต้องเคยปรากฏอย่างน้อย 20 ครั้งถึงจะอยู่ในตารางไร
                                                                                                                                          # sublinear_tf ต้องการปรับขนาดหรือไม่ (True,False)
tfidf_fit = tfidf.fit(train_data_df.texts) #คำสั่ง tfidf.fit(train_data_df.texts) จะทำการฝึก (fit) TfidfVectorizer ด้วยข้อมูลที่อยู่ใน train_data_df.texts |||**ซึ่งจะทำให้เกิดการคำนวณค่า TF-IDF**||| สำหรับคำแต่ละคำในเวกเตอร์ของแต่ละเอกสารในข้อมูลอบรม (training data)
text_train = tfidf_fit.transform(train_set_df.texts) #text_train = tfidf_fit.transform(train_set_df.texts): คำสั่งนี้ใช้ tfidf_fit ที่ได้จากขั้นตอนก่อนหน้าเพื่อแปลงข้อมูลข้อความในคอลัมน์ texts ของ DataFrame train_set_df เป็นเวกเตอร์ TF-IDF โดยผลลัพธ์จะได้เป็น text_train
text_val = tfidf_fit.transform(val_set_df.texts)
text_test = tfidf_fit.transform(test_data_df.test)


##### นำ feature ที่ได้จากตาราง TF-IDF ไว้ในตัวแปล #####

x_train = text_train.toarray()
x_val = text_val.toarray()
x_test = text_test.toarray()

y_train = train_set_df.categories
y_val = val_set_df.categories

##################################################################################################################### สร้างโมเดล ################################################################################################
# สร้างโมเดล Logistic Regression โดยกำหนดพารามิเตอร์ดังนี้
# C=2: ค่า regularization parameter ซึ่งใช้ในการควบคุมการลดความผิดพลาดและป้องกันการ overfitting
# penalty="l2": ใช้ l2 regularization เพื่อลด overfitting โดยให้ความสำคัญเพิ่มขึ้นเมื่อค่าพารามิเตอร์มีค่าใหญ่
# solver="liblinear": เลือก solver ในการหาค่าสูงสุดของฟังก์ชันเชิงเส้น ที่เหมาะสมกับข้อมูลที่มีขนาดเล็กถึงปานกลาง
# dual=False: ตั้งค่า dual parameter เป็น False เพื่อลดการคำนวณที่ไม่จำเป็นในกรณีที่ solver="liblinear"
# multi_class="ovr": ใช้โหมด one-vs-rest (OvR) ในการจัดการกับหลายคลาสที่มีการจัดแบ่งที่เป็นไปได้แตกต่างกัน


##  /*** หรับ Logistic Regression จะใช้ค่า C (regularization parameter) เพื่อควบคุมความซับซ้อนของโมเดลแทนการกำหนด epoch โดยตรง 
##  /*** ซึ่ง C มีหน้าที่ในการปรับความเข้มของ regularization โดยที่ค่า C มากจะทำให้โมเดลเรียนรู้จากข้อมูล training ได้มากขึ้น และค่า C น้อยจะทำให้โมเดลเรียนรู้จากข้อมูล training ได้น้อยลง


model = sklearn.linear_model.LogisticRegression(C=2,penalty="l2",solver="liblinear",dual=False,multi_class="ovr")
model.fit(x_train,y_train) ## ทำการฝึกโมเดล Logistic Regression ด้วยข้อมูล x_train, y_train
model.score(x_val,y_val)   ## คำนวณคะแนนประสิทธิภาพของโมเดลบนข้อมูล validation set (x_val, y_val)


# evaluate model

y_val_pre = model.predict(x_val)
print(sklearn.metrics.classification_report(y_pred=y_val_pre,y_true=y_val))

# Save model

model_filename = 'logistic_regression_model.pkl'

joblib.dump(model, model_filename)


####################################################################################################################### test #################################################################################################
loaded_model = joblib.load('logistic_regression_model.pkl')
sample1 = ["ร้านนี้ก็อร่อยดีนะ","อาหารร้านนี้ไม่ค่อยอร่อยเลย"]
sample2 = ["ร้านนี้อร่อยตรงไหนว่ะ","อาหารร้านนี้ไม่เคยไม่อร่อยเลย"]

#### ก่อนการทำนายต้องทำการหา feature ก่อน ######## 
sample_feature = tfidf_fit.transform([sample2[0]]) 

pred= loaded_model.predict(sample_feature)

print(pred)
