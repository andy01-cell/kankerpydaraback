import csv

import os

# Get the list of all files and directories
base_path = "E://dataset deep learning//Stadium kanker payudara//Dataset - weka//"
kanker1 = base_path + "stadium 1"
kanker2 = base_path + "stadium 2"
kanker3 = base_path + "stadium 3"
kanker4 = base_path + "stadium 4"

student_header = ['gambar', 'stadium']

dir_list1 = os.listdir(kanker1)
dir_list2 = os.listdir(kanker2)
dir_list3 = os.listdir(kanker3)
dir_list4 = os.listdir(kanker4)

# prints all files

x = []
stadium1=[]
stadium2=[]
stadium3=[]
stadium4=[]

for i in range(0, len(dir_list1)):
    stadium1 = [dir_list1[i],1]
    x.append(stadium1)

for i in range(0, len(dir_list2)):
    stadium2 = [dir_list2[i],2]
    x.append(stadium2)

for i in range(0, len(dir_list3)):
    stadium3 = [dir_list3[i],3]
    x.append(stadium3)

for i in range(0, len(dir_list4)):
    stadium4 = [dir_list4[i],4]
    x.append(stadium4)


with open('kankerpydara.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(student_header)
    # Use writerows() not writerow()
    writer.writerows(x)