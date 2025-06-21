list_store = []

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

add_element(2, 3)

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(98, 12, 78) * 10

result = calculate(98, 12, 78)
print(result * 10)

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge

calculate(98, 12, 78) * 10

def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output

type(calculate(98, 12, 78))
varm, moisture, charge, output = calculate(98, 12, 78)

def inner_function():
    print("İç fonksiyon çalışıyor.")

def outer_function():
    print("Ana fonksiyon çalışıyor.")
    inner_function()  # iç fonksiyon çağrısı

outer_function()

def calculate(warm, moisture, charge):
    return int((warm + moisture) / charge) 

def standardization(a, p):
    return a * 10 / 100 * p * p

def all_calculation(warm, moisture, charge, p):
    a = calculate(warm, moisture, charge)
    b = standardization(a, p)
    return b

all_calculation(1, 9, 10, 12)

all_calculation(1, 3, 5, 19, 12)

all_calculation(1, 35, 5, 12)

def my_function():
    global local_variable  # Global değişken tanımlama
    local_variable = 10
    print(local_variable)

    my_function()  # Lokal değişken, sadece fonksiyon içinde erişilebilir
    print(local_variable)  # Lokal değişken, fonksiyon dışında erişilemez

global_variable = 20
def another_function():
    print(global_variable)  # Global değişken, fonksiyon içinde erişilebilir

another_function()  # Global değişkene erişilebilir
print(global_variable)  # Global değişkenin değeri erişilebilir

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(int(salary*20/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(1500, 10)
new_salary(2000, 20)

salaries = [1000, 2000, 3000, 4000, 5000]
for salary in salaries:
    print(new_salary(salary, 20))

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)
alternating("hi my name is john and i am learning python")

i = 0 

while i < 5:
    print(i)
    i += 1

while True:
    user_input = input("Bir sayı girin (çıkmak için 'q' ya da 'Q' tuşuna basın): ")
    if user_input.lower() == "q":
        break
    else:
        number = int(user_input)
        print("Girilen sayının karesi:", number ** 2)

for salary in salaries:
    if salary == 3000:
        break
    print(salary)


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for number in numbers:
    if number % 2 == 0:
        continue
    print(number)

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

students = ["John", "Mark", "Venessa", "Mariam"]

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

print (f'A: {A}, B: {B}')

students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")

students = ["John", "Mark", "Venessa", "Mariam"]
departments = ["mathematics", "statistics", "physics", "astronomy"]
ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

import seaborn as sns
import pandas as pd

# Load the tips dataset
tips = sns.load_dataset("tips")

tips['total_bill'] = tips[['total_bill', 'tip']].apply(lambda x: x[0] + x[1], axis=1)
tips
tips['total'] = tips['total_bill'] + tips['tip']

(lambda x, y: x + y)(3, 5)

# Import necessary libraries
import seaborn as sns
import pandas as pd

# Load the tips dataset
tips = sns.load_dataset("tips")

# get 3-fold of total_bill
tips['bill_3_fold'] = tips['total_bill'].apply(lambda x: x * 3)
tips.head()

# Bir liste oluşturalım
numbers = [1, 2, 3, 4, 5]

# Her bir elemanın karesini hesaplayan bir fonksiyon
def square(x):
    return x ** 2

# map() fonksiyonunu kullanarak her elemanın karesini hesaplayalım
result = list(map(square, numbers))

# map nesnesini listeye dönüştürelim
result_list = list(result)
print(result_list)

result_list = []

for i in numbers:
    result_list.append(square(i))
print(result_list)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

def is_even(x):
    return x % 2 == 0

filtered_numbers = list(filter(is_even, numbers))
print(filtered_numbers)

from functools import reduce 

numbers = [1, 2, 3, 4, 5] 

# Toplama işlemiyle reduce kullanımı
result = reduce(lambda x, y: x / y, numbers)
print(result)  # Output: 15

numbers = [1, 2, 3, 4, 5]
squares = [num**2 for num in numbers if num % 2 == 0]
print(squares)  # Output: [4, 16]

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

salaries = [1000, 2000, 3000, 4000, 5000]
[new_salary(salary * 2, 15) if salary < 3000 else new_salary(salary, 15) for salary in salaries]

salaries = [1000, 2000, 3000, 4000, 5000]
[salary * 2 for salary in salaries]

salaries = [1000, 2000, 3000, 4000, 5000]
[salary * 2 for salary in salaries if salary < 3000]

salaries = [1000, 2000, 3000, 4000, 5000]
[salary * 3 if salary < 3000 else salary * 2 for salary in salaries]

salaries = [1000, 2000, 3000, 4000, 5000] 

def new_salary(x):
    return x * 20 / 100 + x 

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

numbers = [1, 2, 3, 4, 5]
squared_dict = {x: x**2 for x in numbers}
print(squared_dict)

import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')

# Identify categorical columns
[col for col in df.columns if df[col].dtype == 'object']
# Identify numerical columns
[col for col in df.columns if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10]
# Identify boolean columns
[col for col in df.columns if df[col].dtype == 'bool']
# Identify datetime columns
[col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
# Identify columns with missing values
[col for col in df.columns if df[col].isnull().any()]
# Identify columns with a specific data type
[col for col in df.columns if df[col].dtype == 'category']
# Identify columns with a specific data type
[col for col in df.columns if df[col].dtype == 'int64']
# Identify columns with a specific data type
[col for col in df.columns if df[col].dtype == 'float64']
# Identify columns with a specific data type
[col for col in df.columns if df[col].dtype == 'object']


dictionary = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

{k: v ** 2 for (k, v) in dictionary.items()}
# Output: {'a': 1, 'b': 4, 'c': 9, 'd': 16}

{k.upper(): v for (k, v) in dictionary.items()}
# Output: {'A': 1, 'B': 2, 'C': 3, 'D': 4}

{k.upper(): v*2 for (k, v) in dictionary.items()}
# Output: {'A': 2, 'B': 4, 'C': 6, 'D': 8}

numbers = range(10)
{n: n ** 2 for n in numbers if n % 2 == 0}

import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]
df.columns

df = sns.load_dataset("car_crashes")
df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]
df.columns

import seaborn as sns

df = sns.load_dataset("car_crashes")

num_cols = [col for col in df.columns if df[col].dtype != "O"]
agg_list = ["mean", "min", "max", "sum", "var", "std", "count", "median", "skew", "kurtosis"]
new_dict = {col: agg_list for col in num_cols}
print(new_dict)
df[num_cols].head()
df[num_cols].agg(new_dict)

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4,"String",3.2, False]
d = {"Name": "Jake", "Age": [27,56], "Adress": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science","Python"}

type_list = [x, y, z, a, b, c, l, d, t, s]
types = [print(f"{item}: {type(item)}") for item in type_list]

lst = ["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.

data_list = lst[0:4]
data_list

# Adım 4: Sekizinci index'teki elemanı silin.

lst.pop(8)
lst

# Adım 5: Yeni bir eleman ekleyin.

lst.append(99)
lst


# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.

lst.insert(8, "N")
lst


###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America",18],
        'Daisy':["England",12],
        'Antonio':["Spain",22],
        'Dante':["Italy",25]}


# Adım 1: Key değerlerine erişiniz.

dict.keys()

# Adım 2: Value'lara erişiniz.

dict.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict.update({"Daisy": ["England",13]})
dict

dict["Daisy"][1] = 14
dict


# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Adım 5: Antonio'yu dictionary'den siliniz.

dict.pop("Antonio")
dict



###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atıyan ve bu listeleri return eden fonskiyon yazınız.
###############################################

lst = [2,13,18,93,22]

def func(list):

    çift_list = []
    tek_list = []

    for i in list:
        if i % 2 == 0:
            çift_list.append(i)
        else:
            tek_list.append(i)

    return çift_list, tek_list


çift, tek = func(lst)
print("Çift Sayılar:", çift)
print("Tek Sayılar:", tek)

#List comp. çözümü.
çift_list = [i for i in lst if i % 2 == 0]
tek_list = [i for i in lst if i % 2 != 0]

def tek_cift_list(lst):
    return [i for i in lst if i % 2 == 0], [i for i in lst if i % 2 != 0]

çift_list, tek_list = tek_cift_list(lst)
print("Çift Sayılar:", çift_list)
print("Tek Sayılar:", tek_list)



###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]

for i,x in enumerate(ogrenciler):
    if i<3:
        i += 1
        print(f'Mühendislik Fakültesi {i}. öğrenci: {x}')
    else:
        i -= 2
        print(f'Tıp Fakültesi {i}. öğrenci: {x}')

# Alternatif çözüm
for i, student in enumerate(ogrenciler):
    if i < 3:
        print(f"Mühendislik Fakültesi {i + 1}. öğrenci: {student}")
    else:
        print(f"Tıp Fakültesi {i - 2}. öğrenci: {student}")

# Alternatif çözüm 2
for i, student in enumerate(ogrenciler):
    faculty = "Mühendislik Fakültesi" if i < 3 else "Tıp Fakültesi"
    student_number = i + 1 if i < 3 else i - 2
    print(f"{faculty} {student_number}. öğrenci: {student}")

# List comprehension ile alternatif çözüm 3 
ogrenci_listesi = [f"Mühendislik Fakültesi {i + 1}. öğrenci: {student}" if i < 3 else f"Tıp Fakültesi {i - 2}. öğrenci: {student}" for i, student in enumerate(ogrenciler)]
for ogrenci in ogrenci_listesi:
    print(ogrenci)

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]


for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
  print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")


###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])
kume1.intersection(kume2)

def kume(set1,set2):
    if set1.issuperset(set2):
        print(f'Ortak Elemanlar: {set1.intersection(set2)}')
    else:
        print(f"")

kume(kume1,kume2)

def kume_kontrol(kume1, kume2):
    if kume1.issuperset(kume2):  # kume1, kume2'yi kapsıyor mu?
        ortak = kume1.intersection(kume2)
        print("Ortak elemanlar:", ortak)
    else:
        fark = kume2.difference(kume1)
        print("2. kümenin 1. kümeden farkı:", fark)

# Test
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

kume_kontrol(kume1, kume2)

import numpy as np

# [1, 2, 3, 4, 5] elemanlarından oluşan bir NumPy array oluşturma
arr1 = np.array([1, 2, 3, 4, 5])

# Oluşturulan array'in veri tipini kontrol etme
type(np.array([1, 2, 3, 4, 5]))
# Çıktı: <class 'numpy.ndarray'>

import numpy as np

arr = np.array([1, 2, 3, 4, 5])

arr.argmax()
arr.argmin()
arr.any()
arr.astype('int')
arr.ndim
arr.nbytes
arr.shape
arr.size
arr.dtype
arr.itemsize
arr.data

ar = np.random.randint(1, 10, size=9)
ar

ar.reshape(3, 3)

import numpy as np

a = np.random.randint(10, size=10)

a[0]
a[0:5]
a[0] = 999

import numpy as np

# 3 satır ve 5 sütundan oluşan ve elemanları 0 ile 9 arasında rastgele sayılardan oluşan bir NumPy dizisi oluşturma
m = np.random.randint(10, size=(3, 5))
# Dizinin ilk satırı ve ilk sütununun elemanını seçme
m[0, 0]

# Dizinin ikinci satırın ve ikinci sütununun elemanını seçer.
m[1, 1]

# Dizinin üçüncü satırın ve dördüncü sütununun elemanını seçer.
m[2, 3]

# m dizisinin üçüncü satırın ve dördüncü sütununun elemanına 999 değerini atar.
m[2, 3] = 999

# m dizisinin üçüncü satırın ve dördüncü sütununun elemanına 2.9 değerini atar. Ancak, m bir tamsayı dizisi olduğu için bu atama işlemi sonucunda değer 2 olarak kaydedilir.
m[2, 3] = 2.9

# m dizisinin tüm satırlarını ve ilk sütununu seçer.
m[:, 0]

# m dizisinin ikinci satırını ve tüm sütunlarını seçer.
m[1, :]

# m dizisinin ilk iki satırını ve ilk üç sütununu seçer.
m[0:2, 0:3]

arr = np.array([10, 20, 30, 40, 50])

indices = [1, 3]  # Fancy Index olarak kullanılacak liste

selected_elements = arr[indices]  # Fancy Index ile elemanları seçme

print(selected_elements)

mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mat

indices = np.array([[0, 2], [1, 2]])  # Fancy Index olarak kullanılacak matris

selected_elements = mat[indices]  # Fancy Index ile elemanları seçme

selected_elements

x = -2

np.abs(x)

x**2

# Denklem sistemini tanımlama
# 2x + y = 10
# x + 3y = 15

coefficients = np.array([[2, 1], [1, 3]])
constants = np.array([10, 15])

# Denklem sistemini çözme
solution = np.linalg.solve(coefficients, constants)
print("x =", solution[0])
print("y =", solution[1])

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)