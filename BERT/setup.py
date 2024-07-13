import sklearn.model_selection
import process_text
import pandas as pd
import os


##########################################################################  setup data  ###########################################################################################

train_path = os.path.join(r".\data\wisesight-sentiment-master\kaggle-competition\train.txt")
train_label_path = os.path.join(r".\data\wisesight-sentiment-master\kaggle-competition\train_label.txt")
test_path = os.path.join(r".\data\wisesight-sentiment-master\kaggle-competition\test.txt")
test_label_path = os.path.join(r".\data\wisesight-sentiment-master\kaggle-competition\test_label.txt")

with open(train_path,"r",encoding="utf-8") as f:   #with open('train.txt') as f:: เปิดไฟล์ train.txt สำหรับการอ่านโดยใช้คำสั่ง with ซึ่งจะทำการเปิดไฟล์และใช้งานได้อัตโนมัติ 
                                                   # utf-8 ซึ่งเป็นการเข้ารหัสทั่วไปที่รองรับอักขระหลากหลายภาษา:
    texts = [line.strip() for line in f.readlines() ] #f.readlines() เพื่ออ่านข้อมูลจากไฟล์ทั้งหมดและแบ่งแยกเป็นบรรทัดๆ

with open(train_label_path,mode="r",encoding= "utf-8") as f:
    categories = [line.strip() for line in f.readlines()]

with open(file=test_path,mode="r",encoding="utf-8") as f:
    tests = [line.strip() for line in f.readlines()]

with open(file=test_label_path,mode="r",encoding="utf-8") as f:
    test_label = [line.strip() for line in f.readlines()]

train_data = {"texts":texts,
        "categories":categories}
train_data_df = pd.DataFrame(train_data)

train_data_df["processed"] = train_data_df["texts"].map(lambda x : "|".join(process_text.process_text(x))) #"|".join(...): ใช้ join() เพื่อนำรายการ (list) ของคำที่ได้จาก process_text มารวมกันเป็นสตริงเดียว โดยใช้เครื่องหมาย "|" เป็นตัวคั่นระหว่างคำแต่ละคำ
train_data_df["wc"] = train_data_df.apply(lambda x : len(x.processed.split("|")),axis=1)#.split("|") เป็นคำสั่งที่ใช้ในการแบ่งสตริง processed ออกเป็นรายการ (list) ของคำ โดยใช้เครื่องหมาย "|" เป็นตัวคั่นในการแบ่ง. --/// 
                                                                                        #จำนวนคำทั้งหมดที่แบ่งได้มีจำนวนเท่าไร
train_data_df["uwc"] = train_data_df.apply(lambda x : len(set(x.processed.split("|"))),axis=1) #จำนวนคำทั้งหมดที่แบ่งได้มีจำนวนเท่าไรถ้าไม่นับคำซ้ำ


test_data = {"test":tests,
             "categories":test_label}
test_data_df =pd.DataFrame(test_data)

test_data_df["processed"] = test_data_df["test"].map(lambda x : "|".join(process_text.process_text(x))) #"|".join(...): ใช้ join() เพื่อนำรายการ (list) ของคำที่ได้จาก process_text มารวมกันเป็นสตริงเดียว โดยใช้เครื่องหมาย "|" เป็นตัวคั่นระหว่างคำแต่ละคำ
test_data_df["wc"] = test_data_df.apply(lambda x : len(x.processed.split("|")),axis=1)#เป็นคำสั่งที่ใช้ในการแบ่งสตริง processed ออกเป็นรายการ (list) ของคำ โดยใช้เครื่องหมาย "|" เป็นตัวคั่นในการแบ่ง.
test_data_df["uwc"] = test_data_df.apply(lambda x : len(set(x.processed.split("|"))),axis=1)

train_data_df.to_csv("train_df.csv")
test_data_df.to_csv("test_df.csv")



