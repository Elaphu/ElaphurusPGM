import pandas as pd
data = pd.DataFrame(data={
                          'fruit': ["banana", "apple", "banana", "apple", "banana","apple", "banana", 
                                    "apple", "apple", "apple", "banana", "banana", "apple", "banana",], 
                          'tasty': ["yes", "no", "yes", "yes", "yes", "yes", "yes", 
                                    "yes", "yes", "yes", "yes", "no", "no", "no"],
                          # 'weight': ["heavy","light","heavy","heavy","light","heavy","heavy","heavy",
                                     # "light","heavy","light","heavy","heavy","light"],
                          'color' : ["yellow","red","yellow","red","yellow","red","yellow",
                                     "red","red","red","yellow","yellow","red","yellow"],
                          'size': ["large", "large", "large", "small", "large", "large", "large",
                                    "small", "large", "large", "large", "large", "small", "small"]
                            })
print(data)


import numpy as np
from ElaphurusPGM.Model import BayesianModel
from ElaphurusPGM.BicScore import Bic
from ElaphurusPGM.MaxLikelihood import MaxLikelihood
from ElaphurusPGM.ExhaustiveSearch import ExhaustiveSearch

ES = ExhaustiveSearch(data)
model = ES.estimate()
print("best structure", model.edges())
MLE = MaxLikelihood(model, data)
model.add_cpds(MLE.get_parameters())
for cpd in model.get_cpds():
    print(cpd.values)
import pickle
with open("fruit.pickle",'wb') as f:
    pickle.dump(obj=model,file=f)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
from matplotlib.font_manager import *
import networkx as nx
from matplotlib import pyplot as plt
nx.draw(best_model, with_labels=True,node_size=1600, cmap=plt.cm.Blues,
        node_color=range(len(best_model)),
        prog='dot')
plt.show()