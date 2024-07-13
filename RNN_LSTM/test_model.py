######################################################################  test RNN_LSTM.py  #############################################################################
import tensorflow as tf
import numpy as np
# ระบุพาธที่มีไฟล์ saved_model.pb อยู่
saved_model_path = r'D:\machine_learning_AI_Builders\บท4\NLP\checkpoint'

# โหลด SavedModel ด้วย Keras
loaded_model = tf.keras.models.load_model(saved_model_path)
model = loaded_model
# ทำนายโดยใช้โมเดลที่โหลดมา
# ต้องทราบ input shape ของโมเดลที่เก็บไว้ใน saved_model.pb
# ตัวอย่างการใช้งาน:
# predictions = loaded_model.predict(input_data)
y_class = ['neg', 'neu', 'pos', 'q']
sample_text = ['ร้านอาหารร้านนี้อร่อยมาก',"ฉันไม่อยากพบเธออีก"]
sample_text = [i for i in sample_text]


predictions = model.predict(np.array(sample_text))
result = [y_class[i] for  i in predictions.argmax(axis=1)]
[print(f"{sample_text[i]}:{result[i]}") for i,_ in enumerate(result)]
# sample_text = ('ร้านอาหารร้านนี้อร่อยมาก')


