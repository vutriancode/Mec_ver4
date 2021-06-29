import random as rd
import numpy as np
from  numpy.random import poisson as ps
from pathlib import Path
import os
import sys
path =os.path.abspath(__file__)
path =Path(path).parent.parent
def random_task_type_1(number_task):
    for i in range(100):
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_1_test",i),"w") as output:
            indexs=ps(number_task)
            m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
            m1 = np.random.randint(1000,2000,indexs)
            m2 = np.random.randint(100,200,indexs)
            m3 = np.random.randint(500,1500,indexs)
            m4 = 1+np.random.rand(indexs)*2
            for j in range(indexs):
                output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    for i in range(100):
        with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_1_train",i),"w") as output:
            indexs=rd.randint(900,1200)
            m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
            m1 = np.random.randint(1000,2000,indexs)
            m2 = np.random.randint(100,200,indexs)
            m3 = np.random.randint(500,1500,indexs)
            m4 = 1+np.random.rand(indexs)*2
            for j in range(indexs):
                output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))


def random_task_type_2():
    for k in range(2):
        for z in range(50):
            with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_2_test",k*50+z),"w") as output:
                if k == 0:
                    indexs = 1000+z*5 
                else:
                    indexs = 1250-z*5 
                m = np.sort(np.random.randint((50*k+z)*300,(50*k+z+1)*300,indexs))
                m1 = np.random.randint(1000,2000,indexs)
                m2 = np.random.randint(100,200,indexs)
                m3 = np.random.randint(500,1500,indexs)
                m4 = 1+np.random.rand(indexs)*2
                for j in range(indexs):
                    output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))
    for k in range(2):
        for z in range(50):
            with open("{}/data_task/{}/datatask{}.csv".format(str(path),"data_2_train",k*50+z),"w") as output:
                if k == 0:
                    indexs = 1000+z*5 
                else:
                    indexs = 1250-z*5 
                m = np.sort(np.random.randint((50*k+z)*300,(50*k+z+1)*300,indexs))
                m1 = np.random.randint(1000,2000,indexs)
                m2 = np.random.randint(100,200,indexs)
                m3 = np.random.randint(500,1500,indexs)
                m4 = 1+np.random.rand(indexs)*2
                for j in range(indexs):
                    output.write("{},{},{},{},{}\n".format(m[j],m3[j],m1[j],m2[j],m4[j]))

if __name__=="__main__":
    types = 1
    if len(sys.argv) > 1:
        types = int (sys.argv[1])
    if types ==1:
        random_task_type_1(1100)
    elif types==2:
        random_task_type_2()
    else:
        print("vui lòng chọn kiểu dữ liệu type 1 or 2")