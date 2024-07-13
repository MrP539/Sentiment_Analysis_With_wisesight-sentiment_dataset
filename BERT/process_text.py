import re
import emoji
import pythainlp

################################################################################ process text ####################################################################################################################################

# กำหนดรูปแบบ URL ให้แทนด้วย xxurl
def replace_url(text):  
    URL_PATTERN = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
    #แทนที่ URL ด้วย 'xxurl' ในข้อความ
    return re.sub(URL_PATTERN, 'xxurl', text)

#input  -> "Hello! Visit https://example.org.th for more info. 😊😊"
#output -> "Hello! Visit xxurl                  for more info. 😊😊"


# จะทำการค้นหาและแทนที่ตัวอักษรที่ซ้ำกันอย่างน้อย 3 ตัวติดต่อกันในข้อความ text ด้วยข้อความที่ได้จากการเรียกใช้ _replace_rep()
def relpace_rep(text):
    def replace_rep_(m):#ป็นฟังก์ชันที่ใช้กำหนดรูปแบบในการแทนที่ข้อความที่ตรงกับ pattern นี้ โดยที่ m คือ object ที่เกิดจากการตรงกับ pattern นั้น ๆ และฟังก์ชันนี้จะคืนข้อความที่ใช้แทนที่ในที่นี้เป็น f"{c}xxrep"
        #print(m)
        c,cc = m.groups() #.groups() เป็นเมธอดที่ใช้ในการดึงข้อมูลที่ตรงกับกลุ่ม (group) ในการตรงกับรูปแบบของ regular expression 
        return f"{c}xxrep"
    re_rep = re.compile(r"(\S)(\1{2,})")

    return re_rep.sub(replace_rep_,text) #.sub() ใช้เพื่อค้นหาตำแหน่งที่ตรงกับรูปแบบ r"(\S)(\1{2,})" ซึ่งหมายถึงตัวอักษรที่ไม่ใช่ช่องว่าง (\S) ตามด้วยตัวอักษรที่ซ้ำกันอย่างน้อย 2 ตัว (\1{2,})

#input  -> "Hello! Visit https://example.org.th for more infooo. 😊😊"
#output -> Hello! Visit https://example.org.th for more infoxxrep. 😊😊"

def ungroup_emoji(toks):
    res = []
    for tok in toks:
        if emoji.emoji_count(tok) == len(tok):
            for char in tok:
                res.append(char)
        else:
            res.append(tok)
    return (res)

#input  -> "Hello! Visit https://example.org.th for more infooo. 😊😊"  ** ต้องมีการ แยกข้อความเป็น tokens ก่อน
#output -> ['Hello!', 'Visit', 'https://example.org.th', 'for', 'more', 'infooo.', '😊', '😊']


def process_text(text):
    #pre rules
    res = text.lower().strip()# แปลงเป็นตัวพิมพ์เล็กและลบช่องว่างที่อยู่ที่ด้านหน้าและด้านหลังของข้อความ
    res = replace_url(res)    # กำหนดรูปแบบ URL ให้แทนด้วย xxurl
    res = relpace_rep(res)    # ทำการค้นหาและแทนที่ตัวอักษรที่ซ้ำกันอย่างน้อย 3 ตัวติดต่อกันในข้อความ text ด้วยข้อความที่ได้จากการเรียกใช้ _replace_rep()
    res = [word for word in pythainlp.word_tokenize(res) if word and not re.search(pattern=r"\s+",string=word)]# word_tokenize จากไลบรารี PyThaiNLP ใช้สำหรับการตัดคำภาษาไทยออกเป็นหน่วยคำ (tokenization) เช่น  "ไปเที่ยว" -->  ["ไป", "เที่ยว"]
                                                                                                               # r"\s+" หมายถึงตัวอักษรที่เป็นช่องว่าง (whitespace characters) ทุกชนิด (space, tab, newline) ที่เกิดขึ้นหนึ่งตัวหรือมากกว่าครั้งเดียว Ex "Hello   world\tfrom\nPython" -> ['   ', '\t', '\n']
                                                                                                               # if word: ตรวจสอบว่าตัวแปร word มีค่าหรือไม่ ซึ่งหาก word มีค่า (empty string) จึงเป็น True ในเงื่อนไขนี้
                                                                                                               # not re.search(pattern=r"\s+", string=word): ตรวจสอบว่า word ไม่มี whitespace characters (ช่องว่าง, tab, newline) ใน word
    res = ungroup_emoji(res) #ungroup_emoji
    return res
#input  -> "Hello! Visit https://example.org.th for more infooo. 😊😊"  ** ต้องมีการ แยกข้อความเป็น tokens ก่อน
#output -> ['hello', '!', 'visit', 'xxurl', 'for', 'more', 'infoxxrep', '.', '😊', '😊']



