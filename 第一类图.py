import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
#设置字体显示
from matplotlib.font_manager import FontProperties
myFont=FontProperties(fname=r'C:\WINDOWS\FONTS\SIMKAI.TTF',size=14)
sns.set(font=myFont.get_name())

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

np.set_printoptions(threshold=1e6)

# 画图函数
def plot(plot_data1,plot_data2, name1, name2,date1,id):
    # 开始画图
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in date1]



    # 画图
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor='white')
    # 左轴
    ax1.plot(x, plot_data1, color='green', label="{}".format(name1))
    ax1.set_xlabel('日期')
    ax1.set_ylabel('{}'.format(name1),fontdict={'weight': 'normal', 'size': 15})
#    ax1.set_ylim(0, 5)
    ax1.legend(loc='upper left')
    # 右轴
    ax2 = ax1.twinx()
    ax2.plot(x, plot_data2,  color='red', label='{}'.format(name2))
    ax2.set_ylabel('{}'.format(name2),fontdict={'weight': 'normal', 'size': 15})

    ax2.legend(loc='upper right')
    plt.title('{}指数{}与{}'.format(id,name1,name2))
#    plt.savefig('images2/{0}-{1}-{2}.png'.format(id,name1,name2), format='png')
    #plt.show()
    plt.close()



def plot2(plot_data1, name1, date1,n,title):
    # 开始画图
    date1=date1[:-n]
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in date1]
    plt.title('{}'.format(title))
    plt.plot(x, plot_data1, color='darksalmon', label=name1)


#    plt.legend()  # 显示图例

    plt.xlabel('日期')
    plt.ylabel('{}'.format(name1))
#    plt.show()
#    plt.savefig('images2/{0}.png'.format(name1), format='png')
    plt.close()
    # python 一个折线图绘制多个曲线

def sort(data):
    data1=np.argsort(data)
    data2=np.zeros_like(data1)
    for i in range(len(data2)):
        for j in range(np.size(data2,1)):
            data3=data1[i,j]
            data2[i,data3]=j
    return data2

if __name__ == '__main__':

    # 载入数据
    data1 = pd.read_excel('基础数据/申万指数得分.xlsx',header=0,index_col=0).T
    data2 = pd.read_excel('基础数据/申万指数收盘价.xlsx',header=0,index_col=0)
    data3 = pd.read_excel('基础数据/沪深300得分.xlsx',header=0,index_col=0)
    data4 = pd.read_excel('基础数据/沪深300收盘价.xlsx',header=0,index_col=0)

    date111=data1.columns.values
    date1=date111
    date112=date1[np.linspace(0, 138, 47).astype(int)]
    date2=date112
    data1=data1.T
    data3=data3.T
    data4=data4.T
    value1=data1.values
    value2 = data2.values
    value3 = data3.values
    value4 = data4.values
    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[1:] / value2[:-1]
    return_list2= value4[1:]/value4[:-1]
    aa1=data1.index.values
    aa2=data1.index.values
    for num, fundid in enumerate(data1.columns[0:]):
        print("正在读取第{0}个数据({1})".format(num+1 , fundid))
        score_value = data1[fundid].to_numpy()

        close_value = data2[fundid].to_numpy()
        plot(score_value, close_value, '得分', '收盘价', date1, fundid)
        chaoe=score_value-value3
        adjusted_value=value2.T
        close_value2=np.ravel(adjusted_value[num])
        print(close_value2)
        xiangdui=np.true_divide(close_value2,value4)
        print(xiangdui)
        chaoe=np.ravel(chaoe)
        xiangdui=np.ravel(xiangdui)

        aa1=np.column_stack((aa1,chaoe))
        aa2 = np.column_stack((aa2, xiangdui))
        plot(chaoe, xiangdui, '超额得分', '相对表现', date1, fundid)
    name1=data1.columns.values

    print(aa1)
    aa1=np.delete(aa1,0,axis=1)
    aa2=np.delete(aa2,0,axis=1)
    print(len(aa1))
    pd.DataFrame(aa1,index=date1,columns=data1.columns.values).to_excel(r'第一类图/超额得分.xlsx')
    pd.DataFrame(aa2,index=date1, columns=data1.columns.values).to_excel(r'第一类图/相对表现.xlsx')


