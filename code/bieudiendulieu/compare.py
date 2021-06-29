import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

a=[]
b=[]
for i in range(1,2):
    files=pd.read_csv("result/MAB2/MAB_5phut_s4.csv")
    x=files["mean_reward"].to_numpy()[0:100]
    b.append(x)
    files=pd.read_csv("result/DQN2/reward_5phut_env_s4.csv")
    xx=files["mean_reward"].to_numpy()[0:100]
    a.append(xx)
    #print(a)
files=pd.read_csv("result/DDQN2/reward_5phut_env_s4.csv")
files1=pd.read_csv("result/DuelingDQN2/reward_5phut_env_s4.csv")
#files=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 1/ketqua_oneday.csv")
#files1=pd.read_csv("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/deep q learning 2/ketqua_oneday.csv")
m=[i for i in range(1,101)]
#files=pd.read_excel("C:/Users/vutri/OneDrive/Desktop/15092020/code/result4/compare.xlsx")
fig, ax = plt.subplots()
x=[i for i in range(0,100)]
labels=["an","an1","aN1","an3"]

ax.plot(x,np.average(a,axis=0)[0:100] ,marker='^', markevery=5,label="DQN",color="orange",lw=1)
ax.plot(x, files1["mean_reward"][0:100],marker="P", markevery=5,color="red",label="Dueling DQN",lw=1)
ax.plot(x, files["mean_reward"][0:100],marker='o', markevery=5,label="DDQN",color="blue",lw=1)
ax.plot(x, np.average(b,axis=0)[0:100],marker="*", markevery=5,color="green",label="MAB",lw=1)
ax.set_ylabel('MQoE',fontsize=15)
ax.set_xlabel('Time slots',fontsize=15)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])

plt.setp(ax.get_xticklabels(), fontsize=15)
plt.setp(ax.get_yticklabels(), fontsize=15)
#ax.set_xticklabels(labels)
ax.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), ncol=4)
plt.grid(alpha=0.5)
#loc='upper center'
plt.show()
plt.savefig("Compare_5p.eps")
#print(max(files1["Random"]))
#print(min(files1["Random"]))
#print(np.average(b,axis=0))"""
