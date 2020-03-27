import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0.1,0.5,10)#生成[0.1,0.5]等间隔的十个数据
y=np.exp(x)

error=0.05+0.15*x#误差范围函数

error_range=[error,error]#下置信度和上置信度

plt.errorbar(x,y,yerr=error_range,fmt='o:',ecolor='hotpink',elinewidth=10,ms=10,mfc='wheat',mec='salmon',capsize=10)

plt.xlim(0.05,0.55)#设置x轴显示范围区间
plt.show()