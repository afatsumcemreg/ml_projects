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