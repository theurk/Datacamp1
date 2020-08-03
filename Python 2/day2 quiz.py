#!/usr/bin/env python
# coding: utf-8


#1


a = int(input("Please input number : "))
if a == 0:
    print('เลขศูนย์')
elif a%2 == 0:
    print('เลขคู่')
else:
    print('เลขคี่')


# In[ ]:


#2


a = float(input('Please input first number : '))
b = float(input('Please input second number : '))
while b == 0:
    b = float(input('Please input second number : '))
c = a/b
print('The result is',c)


# In[ ]:


#3


number = 1
lst = []
while number != 0:
    number = int(input('Please input number : '))
    lst.append(number)
from functools import reduce
Sum_lst = reduce(lambda x,y : x+y, lst)
mean = Sum_lst/len(lst)
print(mean)


# In[1]:


#4


a = int(input('Please input number : '))
if a%7 == 0:
    print('หาร 7 ลงตัว')
else:
    print('หาร 7 ไม่ลงตัว')


# In[4]:


#5


a = int(input('Please input number : '))
b = int(input('Please input number : '))
c = int(input('Please input number : '))
if a**2 + b**2 == c**2:
    print('สร้างสามเหลี่ยมมุมฉากได้')
else:
    print('สร้างไม่ได้เด้ออออ')


# In[12]:


#6


# เมื่อกำหนดให้ c เป็นความยาวของด้านที่ยาวที่สุดในรูปสามเหลี่ยม 
c = int(input('Please input the largest number : '))
a = int(input('Please input number : '))
b = int(input('Please input number : '))

while c < a and b:
    c = int(input('Please input the largest number : '))
    a = int(input('Please input number : '))
    b = int(input('Please input number : '))
    
if a**2 + b**2 > c**2:
    print('สามเหลี่ยมมุมแหลม')
elif a**2 + b**2 < c**2:
    print('สามเหลี่ยมมุมป้าน')
elif a**2 + b**2 == c**2:
    print('สามเหลี่ยมมุมฉาก')
print('อิอิ')


# In[9]:


#7



pay = []
for i in range(3):
    a = int(input('Please input number : '))
    if 1000 <= a <= 5000:
        pay.append(a)
    else:
        print('ใส่จำนวนเงินไม่ถูกต้อง ต้องในช่วง 1,000 - 5,000 เท่านั้น')
        break
pay = sum(pay)
if  pay <= 6000:
    pay0 = pay
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายคือ ', pay0, '(ไม่ได้ลดหว่ะ)')
elif 6000 < pay <= 12000:
    pay15 = pay - (pay*15)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายคือ ', pay15, '(ได้ลด 15% อิอิ)')
else:
    12000 < pay
    pay25 = pay - (pay*25)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายคือ ', pay25, '(ได้ลด 25% อิอิ)')


# In[20]:


#8


pay = []
for i in range(4):
    a = int(input('Please input number : '))
    if 1000 <= a <= 5000:
        pay.append(a)
    else:
        print('ใส่จำนวนเงินไม่ถูกต้อง ต้องในช่วง 1,000 - 5,000 เท่านั้น')
        break
pay1_3 = sum(pay[0:3])
pay4 = pay[3]
if  pay1_3 <= 6000:
    pay4_0 = pay4
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_0, '(ไม่ได้ลดหว่ะ)')
elif 6000 < pay1_3 <= 12000:
    pay4_15 = pay4 - (pay4*15)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_15, '(ได้ลด 15% อิอิ)')
else:
    12000 < pay1_3
    pay4_25 = pay4 - (pay4*25)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_25, '(ได้ลด 25% อิอิ)')


# In[ ]:


#9



pay = []
i = 0
while i < 4:
    a = int(input('Please input number : '))
    if 1000 <= a <= 5000:
        pay.append(a)
        i += 1
    else:
        print('ใส่จำนวนเงินไม่ถูกต้อง ต้องในช่วง 1,000 - 5,000 เท่านั้น')
        continue
pay1_3 = sum(pay[0:3])
pay4 = pay[3]
if  pay1_3 <= 4000:
    pay4_0 = pay4
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_0, '(ไม่ได้ลดหว่ะ)')
elif 4000 < pay1_3 <= 9000:
    pay4_25 = pay4 - (pay4*25)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_25, '(ได้ลด 25% อิอิ)')
else:
    9000 < pay1_3
    pay4_30 = pay4 - (pay4*30)/100
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_30, '(ได้ลด 30% อิอิ)')


# In[ ]:


#10



pay = []
i = 0
while i < 4:
    a = int(input('Please input number : '))
    if 1000 <= a <= 5000:
        pay.append(a)
        i += 1
    else:
        print('ใส่จำนวนเงินไม่ถูกต้อง ต้องในช่วง 1,000 - 5,000 เท่านั้น')
        continue
        
credit = input('ลูกค้ามีบัตรเครดิตมั้ยจ้ะ? (ตอบ True/False) : ')

pay1_3 = sum(pay[0:3])
pay4 = pay[3]

if  pay1_3 <= 4000:
    pay4_0 = pay4
    print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_0, '(ไม่ได้ลดหว่ะ)')
elif 4000 < pay1_3 <= 9000:
    if credit != True:
        pay4_25 = pay4 - (pay4*25)/100
        print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_25, '(ได้ลด 25% อิอิ)')
    else:
        pay4_25_credit = (pay4 - (pay4*25)/100) - (pay4*5)/100
        print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_25_credit, '(ได้ลด 25% + 5% อิอิ)')
else:
    9000 < pay1_3
    if credit != 'True':
        pay4_30 = pay4 - (pay4*30)/100
        print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_30, '(ได้ลด 30% อิอิ)')
    else:
        pay4_30_credit = (pay4 - (pay4*30)/100) - (pay4*5)/100
        print('สรุปยอดเงินสุดท้ายที่ต้องจ่ายของมื้อที่ 4 คือ ', pay4_30_credit, '(ได้ลด 30% + 5% อิอิ)')


# In[12]:


#11



String = input('Please input your string : ')
new_String ='' 
  
for a in String: 
    if (a.isupper()) == True: 
        new_String +=(a.lower()) 
    elif (a.islower()) == True: 
        new_String +=(a.upper()) 
#ใส่ไว้ เผื่อเคสที่มีช่องวาง
    elif (a.isspace()) == True: 
        new_String += a 
        
print("new_string : ", new_String) 


# In[ ]:


#12



lst = []
for i in range(4):
    a = input('Please input your string : ')
    lst.append(a)
for i in range(4):
    length = len(lst[i])
    print(length)
lst.sort()
print(lst)


# In[20]:


#เก็บไว้เป็นตัวอย่างการเรียงลำดับจากมากไปน้อย


numbers = [ 1 , 3 , 4 , 2 ] #
numbers.sort(reverse=True)
print (numbers)


# In[15]:


#13


temp = True

while temp:
    Hour = int(input('กรุณาใส่จำนวนชั่วโมง : '))
    if 0 <= Hour <= 60:
        break
    else:
        print('คุณใส่จำนวนชั่วโมงไม่ถูกต้อง ต้องระบุตัวเลขในช่วง 0-60 เท่านั้น')
        continue
        
while temp:    
    Minute = int(input('กรุณาใส่จำนวนนาที : '))
    if 0 <= Minute <= 60:
        break
    else:
        print('คุณใส่จำนวนนาทีไม่ถูกต้อง ต้องระบุตัวเลขในช่วง 0-60 เท่านั้น')
        continue
        
Total_pay = (Hour*150) + (Minute*2)
print('ค่าจอดรถทั้งหมด = ', Total_pay)


# In[9]:


#14



temp = True

while temp:
    Hour = int(input('กรุณาใส่จำนวนชั่วโมง : '))
    if 0 <= Hour <= 60:
        if Hour > 0:
            Hour = Hour - 1
            break
        else: 
            break
    else:
        print('คุณใส่จำนวนชั่วโมงไม่ถูกต้อง ต้องระบุตัวเลขในช่วง 0-60 เท่านั้น')
        continue
        
while temp:
    Minute = int(input('กรุณาใส่จำนวนนาที : '))
    if 0 <= Minute <= 60:
        if Minute > 15:
            Hour = Hour + 1
            break
        else: 
            break
    else:
        print('คุณใส่จำนวนชั่วโมงไม่ถูกต้อง ต้องระบุตัวเลขในช่วง 0-60 เท่านั้น')
        continue

    
Total_pay = Hour * 300
print('ค่าจอดรถทั้งหมด = ', Total_pay)


# In[23]:


#15


Total_Head = 0
while Total_Head < 2:
    Total_Head = int(input('กรุณาใส่จำนวนหัวของสัตว์ทั้งหมด : '))
        
Total_Leg = 0
while Total_Leg < 6:
    Total_Leg = int(input('กรุณาใส่จำนวนขาของสัตว์ทั้งหมด : '))
    if Total_Leg %2 != 0:
        Total_Leg = 0
        continue
    else:
        break

# สมการที่ 1 Total_Leg = (2*Bird) + (4*Cow)

# สมการที่ 2 Total_Head = Bird + Cow

# จำนวนนกทั้งหมดคือ

Bird = ((4*Total_Head) - Total_Leg)/2
print('จำนวนนกทั้งหมด คือ ', Bird, 'ตัว')

# จำนวนวัวทั้งหมดคือ

Cow = Total_Head - Bird
print('จำนวนวัวทั้งหมด คือ ', Cow, 'ตัว')

#ค่าภาษีทั้งหมดที่ต้องจ่าย

Tax = (Bird*150) + (Cow*220)
print('ค่าภาษีที่ต้องจ่าย คือ ', Tax, 'บาท')


# In[140]:
# 
