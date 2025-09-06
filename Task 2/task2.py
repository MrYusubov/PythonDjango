from functools import reduce

import pandas as pd

data = pd.read_excel('dataset.xlsx').to_dict('records')

#ekrana seliqeli cixarmaq ucun
def pretty_print(data):
    for record in data:
        print(f'{record}\n')

#1) Map: 'Name' sütununu hamısını böyük hərflərə çevirərək yeni siyahı yaradın.

# new_upper_list = list(map(lambda x:{**x,'Name': x["Name"].upper()}, data))
# pretty_print(new_upper_list)

#2) Map: 'SalaryAZN' dəyərlərini 10% artımla (round) yeni siyahıda göstərin.

# new_salary_list = list(map(lambda x:{**x,'SalaryAZN': round(x['SalaryAZN']*1.1,10)}, data))
# pretty_print(new_salary_list)

#3) Filter: Yalnız 'City' = 'Baku' olan əməkdaşları seçin (sətir siyahısı).

# city_filter_list = list(filter(lambda x: x['City'] == 'Baku', data))
# pretty_print(city_filter_list)

#4) Filter: 'Age' >= 30 və 'Department' = 'Engineering' olanları çıxarın.

# advanced_filter_list = list(filter(lambda x: x['Age'] >=30 and x['Department']=='Engineering', data))
# pretty_print(advanced_filter_list)

#5) Filter: 'Remote' = True və 'Performance' >= 80 olan sətirləri seçin.

# remote_filter_list = list(filter(lambda x: x['Remote'] ==True and x['Performance']>=80, data))
# pretty_print(remote_filter_list)

#6) Reduce: Bütün 'SalaryAZN' toplamını hesablayın.

# total_salary = reduce(lambda x, y: x + y["SalaryAZN"], data, 0)
# print(f'Total Salary: {total_salary}')

#7) Reduce: 'SalaryAZN' üzrə maksimumu və minimumu tapın.

# max_salary = reduce(lambda x, y: x if x > y["SalaryAZN"] else y["SalaryAZN"], data, 0)
#
# min_salary = reduce(lambda x, y: x if x < y["SalaryAZN"] else y["SalaryAZN"], data, data[0]["SalaryAZN"])
#
# print("Maksimum :", max_salary)
# print("Minimum :", min_salary)

#8) Map+Filter: 'Skills' içində 'python' olan əməkdaşların adlarını böyük hərflərlə qaytarın.

# skills_list= list(map(lambda x:{**x,'Name' : x["Name"].upper()},filter(lambda row: "python" in [skill.strip().lower() for skill in row["Skills"].split(",")],data)))
#
# pretty_print(skills_list)

#9) Map: 'JoinDate' dəyərlərini 'YYYY-MM' formatına salın.

# change_format = list(map(lambda row: {**row, "JoinDate": row["JoinDate"][:7]}, data))
# pretty_print(change_format)

#10) Filter+Reduce: 'City' = 'Baku' olanların 'SalaryAZN' cəmini hesablayın.

# total_salary = reduce(lambda x, y: x + y["SalaryAZN"],filter(lambda x:x['City']=='Baku',data), 0)
# print(total_salary)

#11) Map: 'Performance' balına görə kateqoriya təyin edin: 1-59='Low', 60-79='Medium', 80-100='High'.

# combine_data= list(map(lambda x: {**x, "PerformanceCategory":
#                      "Low" if x["Performance"] <= 59 else
#                      "Medium" if x["Performance"] <= 79 else
#                      "High"},data))
# pretty_print(combine_data)

#12) Filter: 'Skills' içində 'django' VƏ 'docker' olan sətirləri çıxarın.

# django_filter = list(filter(lambda row: "django" and "docker" in [skill.strip().lower() for skill in row["Skills"].split(",")],data))
#
# pretty_print(django_filter)

#13) Map: 'Age' → gələn il üçün yaş (Age+1) siyahısını yaradın.

# ageplus = list(map(lambda x:{ **x ,'Age':x['Age']+1}, data))
# pretty_print(ageplus)

#14) Filter+Map: 'Department' = 'Data' olanların maaşını 15% artırıb yeni siyahı yaradın.

# filterplusmap = list(map(lambda x:{**x,'Salary':x['SalaryAZN']*1.5},filter(lambda x: x['Department']=='Data',data)))
# pretty_print(filterplusmap)
