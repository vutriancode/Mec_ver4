import random as rd
import numpy as np
from pathlib import Path
import os
path =os.path.abspath(__file__)
path =Path(path).parent.parent
for i in range(100):
    with open("{}/{}/datatask{}.csv".format(str(path),"data_task",i),"w") as output:
        indexs=rd.randint(1100,1400)
        m = np.sort(np.random.randint(i*300,(i+1)*300,indexs))
        m1 = np.random.randint(1000,2000,indexs)
        m2 = np.random.randint(100,200,indexs)
        m3 = np.random.randint(500,1500,indexs)
        m4 = np.random.randint(1,3,indexs)
        for j in range(indexs):
            output.write("{},{},{},{},{}\n".format(m[j],m3[j],m2[j],m1[j],m4[j]))
    #import pdb;pdb.set_trace()