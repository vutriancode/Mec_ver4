import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def column_chart(strs):
    labels=["combine fuzzy and deep q","deep q learning","fuzzy","random"]
    fig, ax = plt.subplots()
    x=[i for i in range(0,100)]
    files=pd.read_csv("result/DDQN2/thongso_DDQN_s4.csv")
    ax.plot(x,files["server"],label="LS",marker='^', markevery=5,color="orange",lw=1)
    ax.plot(x,files["bus1"],label="VS1",marker="o", markevery=5,color="blue",lw=1)
    ax.plot(x,files["bus2"],label="VS2",marker="P", markevery=5,color="red",lw=1)
    ax.plot(x,files["bus3"],label="VS3",marker="*", markevery=5,color="green",lw=1)
    #ax.set_title("Fuzzy-Controller in Deep Q learning")
    ax.set_xlabel("Time slots",fontsize=15)
    ax.set_ylabel("Number of tasks",fontsize=15)
    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    ax.set_ylim(0,1100)
    plt.show()
column_chart("combine fuzzy and deep q")