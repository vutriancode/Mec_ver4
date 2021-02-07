import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
a=[]
b=[]
fig, ax = plt.subplots(1)
for i in range(1,6):
    files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/Combine fuzzy and deep q "+str(i)+"/ketqua_oneday.csv")
    x=files["mean_reward"].to_numpy()[0:100]
    a.append(x)
    files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning "+str(i)+"/ketqua_oneday.csv")
    xx=files["mean_reward"].to_numpy()[0:100]
    b.append(xx)
    #print(a)
m=[i for i in range(1,101)]


c=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/fuzzy/fuzzy_150.csv")
d=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/random/fuzzy_150.csv")
print(c)
ax.plot(m,np.average(a,axis=0),marker='^', markevery=10,label='FDQO',color="orange")
ax.plot(m,np.average(b,axis=0),marker='o', markevery=10,label='DQL',color="blue")
ax.fill_between(m, np.max(b,axis=0),np.min(b,axis=0), facecolor='#b9deff', alpha=0.5)

#ax.plot(m,c["Fuzzy Controller"][0:100],marker='P', markevery=10,label='Fuzzy')
#ax.plot(m,d["Random"][0:100],marker='x', markevery=10,label='Random')
ax.fill_between(m, np.max(a,axis=0),np.min(a,axis=0), facecolor='#FFA107', alpha=0.5)
ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)

plt.ylim(0,1)
#ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlabel('Time slots',fontsize=15)
ax.set_ylabel('Average QoE',fontsize=15)
plt.grid(alpha=0.5)
#plt.show()

plt.savefig("variation.pdf")