##   LSTM (Long Short-Term Memory) เป็นแบบจำลองโครงข่ายประสาทเทียมที่ถูกออกแบบมาเพื่อการจดจำและการเรียนรู้จากลำดับของข้อมูลหรือข้อความ 
#   (sequence data) ได้ดีขึ้น โดยเฉพาะในกรณีที่มีลำดับหรือความเชื่อมโยงระหว่างข้อมูลที่ยาวนาน (long-range dependencies) หรือมีการเปลี่ยนแปลงภายในลำดับที่ซับซ้อน (complex temporal dependencies)

## feature : "Embedding layer" (ชั้นฝังข้อมูล) เป็นชั้นที่ใช้ในโครงข่ายประสาทเทียม (Neural Network) สำหรับการประมวลผลข้อมูลที่มีลำดับหรือตำแหน่งเชิงลำดับ 
# หน้าที่หลักของ Embedding Layer คือการแปลงข้อมูลเชิงหมวดหมู่ (categorical data) เช่น คำในประโยค ให้เป็นเวกเตอร์เชิงตัวเลขที่มีมิติที่น้อยกว่าและสามารถใช้งานได้ในการฝึกโมเดลเชิงลึก
# เช่น ข้อความ หรือชุดข้อมูลที่มีลำดับเช่นชุดข้อมูลที่ประกอบด้วยข้อความ ภาพและวิดีโอ

##Word embedding คือการแปลงคำศัพท์ในภาษาต่างๆให้อยู่ใน set ของจำนวนจริง หรือที่เรียกว่า vector
import sklearn.metrics
import sklearn.preprocessing
import tensorflow as tf
import process_text
import pythainlp
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import torch

############################################################################ Predata ####################################################################################

def process_test_rnn(text):
    res = text.lower().strip()
    res = process_text.replace_url(text=text)
    res = process_text.relpace_rep(res)
    
    res = [word for word in pythainlp.word_tokenize(res) if word and not re.search(pattern=r"\s+",string=word)]

    res = process_text.ungroup_emoji(res)

    result = " ".join(res)
    return (result)

train_data_df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\NLP\train_df.csv",encoding="utf-8") 
test_data_df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\NLP\test_df.csv",encoding="utf-8")

print(train_data_df.values.tolist()[0])
#print(test_data_df.columns)

train_set_df,val_set_df = train_test_split(train_data_df,test_size=0.15,random_state=1412,shuffle=True)
train_set_df = train_set_df.reset_index(drop=True) #reset_index: ทำการรีเซ็ต Index ของ DataFrame เพื่อให้ Index เริ่มต้นที่ 0 และเพิ่มคอลัมน์ให้เป็น Index ใหม่
val_set_df = val_set_df.reset_index(drop=True) #drop=True: ตัวเลือกที่บอกให้ลบคอลัมน์ Index เดิมที่ถูกเพิ่มเข้าไป


# แปลง format ข้อมูลให้อยู่ในรูปที่ต้องการ ในที่นี้คือ -> [กันแดด คิว เพลส ตัวใหม่ นี่ คุม มัน ดีจริง,...]

text_train = [" ".join(x.split("|")) for x in train_set_df['processed'].values.tolist()]  #การใช้ .values จะแปลงข้อมูลใน DataFrame ให้เป็น NumPy array ซึ่งเป็นโครงสร้างข้อมูลที่มีประสิทธิภาพในการประมวลผลทางคณิตศาสตร์และวิทยาการคำนวณข้อมูลใน Python 
                                                                                          #.split("|")เป็นคำสั่งที่ใช้ในการแบ่งสตริง processed ออกเป็นรายการ (list) ของคำ โดยใช้เครื่องหมาย "|" เป็นตัวคั่นในการแบ่ง.
text_val =  [" ".join(x.split("|")) for x in val_set_df.processed.values.tolist()] 
# input        -> กันแดด|คิว|เพลส|ตัวใหม่|นี่|คุม|มัน|ดีจริง 
# x.split("|") -> ['กันแดด', 'คิว', 'เพลส', 'ตัวใหม่', 'นี่', 'คุม', 'มัน', 'ดีจริง']
# output       -> กันแดด คิว เพลส ตัวใหม่ นี่ คุม มัน ดีจริง


# # ปรับข้อมูล label ให้อยู่ในรูป one hot เพื่อใช้ crossentropy

y_train = train_set_df.categories
y_val = val_set_df.categories

y_class = ['neg', 'neu', 'pos', 'q']
print(f"class : {y_class}")

le= sklearn.preprocessing.LabelEncoder()
le.fit(y_class)

y_train = le.transform(y_train)
y_val = le.transform(y_val)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)

############################################################################ สร้าง Dict Lookup เพื่อนำไปใช้ใน Embedding  ####################################################################################

## ** การสร้าง Dict Lookup จำเป็นต้องมี vacab_size ก่อน
word_count = []
for sent in text_train:
    for i in sent.split():
        word_count.append(i)

vacab_size = len(set(word_count))


# สร้าง dict_lookup

encoder = tf.keras.layers.TextVectorization(max_tokens=vacab_size,output_mode="int") # ทำหน้าที่แปลงข้อความธรรมดา (text) ให้อยู่ในรูปแบบตัวเลข (numerical representation) 
                                                                    # tf.keras.layers.TextVectorization ไม่ได้มีรูปแบบเป็น dict โดยตรง แต่ภายใน layer นี้จะมี vocabulary ซึ่งเป็น mapping ระหว่างคำ (หรือโทเคน) กับ index ที่เป็นตัวเลข ซึ่งสามารถมองได้ว่ามีลักษณะคล้ายกับ dictionary
encoder.adapt(text_train) # .adapt(text_train) ทำหน้าที่ปรับแต่ง (fit) TextVectorization layer ให้เข้ากับข้อมูล text_train ที่ป้อนเข้าไป 
#vocab = np.array(encoder.get_vocabulary())
config_encoder_key = encoder.get_config().keys()
config_encoder = encoder.get_config()
attribute_encoder = dir(encoder)

#print(vocab[0:20])
# print(vacab_size)

# EX -> หา index ของคำ

    # ex = "ร้านอาหารนี้อร่อย"
    # ex = process_test_rnn(text=ex)
    # enconder_ex = enconder(ex).numpy()

    # print(enconder_ex)

# EX -> หา index ของคำ

    #print([vacab[x] for x in enconder_ex])
    # 
############################################################################################################ Create model   ####################################################################################

model =tf.keras.Sequential([
    encoder, #ทำข้อมูลให้เป็น dictionary lookup

    #Embeding layer ทำหน้าที่ แปลงข้อมูลเชิงหมวดหมู่ (categorical data) เช่น คำในประโยค ให้เป็นเวกเตอร์เชิงตัวเลขที่มีมิติที่น้อยกว่าและสามารถใช้งานได้ในการฝึกโมเดลเชิงลึก
    tf.keras.layers.Embedding(
        input_dim=encoder.vocabulary_size(),
        output_dim=512,
        mask_zero= True
    ),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)), # ทำเวกเตอร์ส่งเข้าไปที่ LSTM เพื่อสร้างคุณลักษณะของ vactot เหลานั้น
    tf.keras.layers.Dense(128,activation="relu"),             #จากนั้นส่งไปให้ fully conected เพื่อทำการ predict
    tf.keras.layers.Dense(y_train.shape[1],activation="softmax")

])


checkpoint_path = r'checkpoint/'

checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min',
                             save_weights_only=False,  # บันทึกโมเดลทั้งหมด (ไม่เฉพาะน้ำหนักเท่านั้น)
                             save_format='tf')  # บันทึกในรูปแบบ TensorFlow SavedModel

model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["accuracy"]
              )
#from tensorflow.keras.callbacks import CSVLogger

csvlogger = tf.keras.callbacks.CSVLogger("log.csv")

model.fit(x=np.array(text_train),y=y_train,validation_data=(np.array(text_val),y_val),epochs=10,verbose=2,callbacks=[csvlogger,checkpoint])
value = model.predict(np.array(text_val))
y_val_pred = np.argmax(value,axis=1)
y_val_true = np.argmax(y_val,axis=1)
print(sklearn.metrics.classification_report(y_val_true, y_val_pred))


sample_text = ['ร้านอาหารร้านนี้อร่อยมาก',"ฉันไม่อยากพบเธออีก"]
sample_text = [i for i in sample_text]


predictions = model.predict(np.array(sample_text))
result = [list(le.classes_)[i] for  i in predictions.argmax(axis=1)]
[print(f"{sample_text[i]}:{result[i]}") for i,_ in enumerate(result)]
# sample_text = ('ร้านอาหารร้านนี้อร่อยมาก')