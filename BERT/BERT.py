import pandas as pd
import datasets

import thai2transformers.metrics
import torch
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer,
                          )
from thai2transformers.tokenizers import (
    ThaiRobertaTokenizer,
    ThaiWordsNewmmTokenizer,
    ThaiWordsSyllableTokenizer,
    FakeSefrCutTokenizer
)





########################################################################### create dataset ##########################################################################################

#สร้างการอ้างอิง เพื่อ set model และ dataset

class Args:  
    model_name = 'airesearch/wangchanberta-base-att-spm-uncased'                # ชื่อโมเดล WangchanBERTa ที่ใช้สำหรับภาษาไทย

    dataset_name_or_path = 'wisesight_sentiment'                                # ชื่อหรือเส้นทางไปยังชุดข้อมูลที่ใช้ในการฝึกและทดสอบโมเดล
    feature_col = 'texts'                                                       # ชื่อคอลัมน์ที่เก็บข้อความหรือเนื้อหาของแต่ละตัวอย่างในชุดข้อมูล
    label_col = 'category'                                                      # ชื่อคอลัมน์ที่เก็บป้ายชื่อหรือป้ายกำกับของแต่ละตัวอย่างในชุดข้อมูล
    output_dir = 'models_wongnai/wangchanberta-base-att-spm-uncased_wongnai'    # ไดเรกทอรีที่ใช้ในการบันทึกโมเดลหลังจากการฝึก
    batch_size = 16                                                             # ขนาดของกลุ่มข้อมูลที่ใช้ในแต่ละครั้งในการฝึกโมเดล
    warmup_percent = 0.1                                                        # ร้อยละของจำนวนขั้นตอนในการฝึกที่จะใช้ในขั้นตอนเริ่มต้นเพื่อป้องกันปัญหาการเรียนรู้ที่ไม่เสถียร
                                                                                    ##"การเรียนรู้ที่ไม่เสถียร" หมายถึง การที่โมเดลเรียนรู้จากข้อมูลและปรับค่าพารามิเตอร์ต่างๆ 
                                                                                    # ในขั้นตอนการฝึกโดยที่มีปัญหาในการเรียนรู้ที่ไม่เสถียรหมายถึงการเปลี่ยนแปลงผลลัพธ์ออกมาที่ไม่คงที่ หรือไม่เสถียรในขั้นตอนการฝึกโมเดล
    learning_rate = 3e-05                                                       # อัตราการเรียนรู้ที่ใช้ในการปรับค่าพารามิเตอร์ของโมเดลในแต่ละขั้นตอน
    num_train_epochs = 1                                                        # จำนวนรอบการฝึกโมเดล
    weight_decay = 0.01                                                         # ค่าที่ใช้ในการลดน้ำหนักของพารามิเตอร์เพื่อป้องกันการเรียนรู้ที่เกินไป(overfit)
    metric_for_best_model = 'f1_micro'                                          # เมตริกที่ใช้ในการเลือกโมเดลที่ดีที่สุดในการฝึก
    seed = 1412                                                                 # ค่าเพื่อกำหนดความสุ่มในการทำงานของโมเดลเพื่อให้ผลลัพธ์สามารถทำซ้ำได้ตรงกัน

Args = Args()

#load dataset

dataset= datasets.load_dataset(Args.dataset_name_or_path)
dataset = dataset.map(lambda x : {"labels":x[Args.label_col]},batched=True) # batched=True หมายถึงการประมวลผลข้อมูลเป็นกลุ่ม (batch) แทนที่จะประมวลผลข้อมูลทีละรายการ 
num_label_train_set = len(set(dataset['train']['labels']))
 
#create tokenizer

# AutoTokenizer เลือกที่สะดวกสำหรับการโหลด tokenizer ที่เหมาะสมกับโมเดล wangchanberta ที่คุณใช้โดยไม่ต้องระบุประเภทของ tokenizer 
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=Args.model_name,model_max_length=416) ## tokenize ตัวนี้จะทำการตัดคำ และ encoder ไปในครั่งเดียว
                                                                                                              ## model_max_length=416 หมายถึงความยาวสูงสุดของลำดับข้อความ (sequence) ที่ tokenizer จะจัดการได้ในกระบวนการประมวลผล โ
                                                                                                              # ดยจำกัดความยาวสูงสุดของ tokenized sequence ให้ไม่เกิน 416 tokens

def encoder_func (data):
    result = tokenizer(data[Args.feature_col],truncation=True) #truncation=True เป็นการกำหนดให้ tokenizer ทำการตัดข้อความที่มีความยาวเกินค่าที่กำหนดใน model_max_length ให้พอดีกับขีดจำกัดนี้
    return (result)

encoded_dataset = dataset.map(encoder_func,batched=True) # นำ ตัวแปร dataset เข้าไปใน encoder_func
                                                         # input_ids จะเป็นการ map ค่า ในการทำ dict lookup

#create model 

# โหลดโมเดล wangchanberta ที่มีการฝึกมาแล้วสำหรับงานจำแนกประเภท
model = AutoModelForSequenceClassification.from_pretrained(Args.model_name,num_labels =  num_label_train_set)#ป็นฟังก์ชันจาก Hugging Face Transformers library ที่ใช้ในการโหลดโมเดล ที่เหมาะสมโดยอัตโนมัติตามชื่อ ในท พร้อมกับกำหนดจำนวน label สำหรับการจำแนกประเภท (classification).


#สร้าง การอ้างอิงการtrain โดยเราจะกำหนดให้ตรงกับการอ้างอิงที่สร้างขึ้นก่อนน้า

train_args = TrainingArguments(
    output_dir = Args.output_dir,
    evaluation_strategy = "epoch",
    learning_rate=Args.learning_rate,
    per_device_train_batch_size=Args.batch_size,
    per_device_eval_batch_size=Args.batch_size,
    num_train_epochs=Args.num_train_epochs,
    warmup_steps = int(len(encoded_dataset['train']) * Args.num_train_epochs // Args.batch_size * Args.warmup_percent),
    weight_decay=Args.weight_decay,
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model=Args.metric_for_best_model,
    seed = Args.seed
)

trainer = Trainer(
    model,
    train_args,
    train_dataset = encoded_dataset["train"],
    eval_dataset =  encoded_dataset["validation"],
    tokenizer = tokenizer,
    compute_metrics = thai2transformers.metrics.classification_metrics
)

trainer.train()


pred = trainer.predict(encoded_dataset["validation"])
log = pd.DataFrame.from_dict(pred[2],orient='index').transpose() # ทำหน้าที่สร้าง DataFrame จากพจนานุกรม (dictionary) ที่ถูกส่งเข้ามาผ่าน preds[2] โดยการใช้ค่าพิกัด (orient) ที่กำหนดไว้เป็น 'index'
                                                           # ซึ่งหมายถึงให้ index เป็นค่าใน key ของ dictionary และ columns เป็นค่าที่อยู่ในแต่ละ key ของ dictionary นั้นๆ
                                                           #transpose() ทำให้แถวเป็นคอลัมน์และคอลัมน์เป็นแถว

                                                           # ดังนั้นจึงจะได้ DataFrame ที่มีโครงสร้างเป็นแถวเดียว ดยแต่ละคอลัมน์จะมีชื่อเป็น key ของ dictionary และมีค่าเป็นค่าที่อยู่ในแต่ละ key ของ dictionary นั้นๆ
#EX 

#pred[2] = {'eval_loss': 1.3524476289749146, 'eval_accuracy': 0.3560732113144759, 'eval_f1_micro': 0.3560732113144759, 'eval_precision_micro': 0.3560732113144759, 
    #'eval_recall_micro': 0.3560732113144759, 'eval_f1_macro': 0.23294014214516306, 'eval_precision_macro': 0.24402040882268372, 'eval_recall_macro': 0.23898535490217715, 'eval_nb_samples': 2404}

#                                 0
# eval_loss                1.352448
# eval_accuracy            0.356073
# eval_f1_micro            0.356073
# eval_precision_micro     0.356073    ------->>>      eval_loss  eval_accuracy  eval_f1_micro  eval_precision_micro  eval_recall_micro  eval_f1_macro  eval_precision_macro  eval_recall_macro  eval_nb_samples
#                                                    0  1.352448       0.356073       0.356073              0.356073           0.356073        0.23294               0.24402           0.238985           2404.0
# eval_recall_micro        0.356073
# eval_f1_macro            0.232940
# eval_precision_macro     0.244020
# eval_recall_macro        0.238985
# eval_nb_samples       2404.000000

print(log)

