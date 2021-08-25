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
def plot(plot_data1,plot_data2, name1, name2,date1):
    # 开始画图
    x = [datetime.strptime(str(d), '%Y%m%d').date() for d in date1]
    plt.title('策略回报')
    plt.plot(x, plot_data1, color='green', label=name1)
    plt.plot(x, plot_data2, color='red', label=name2)

    plt.legend()  # 显示图例

    plt.xlabel('日期')
    plt.ylabel('净值')
    #    plt.show()
#    plt.savefig('images/{0}-{1}.png'.format(name1, name2), format='png')
    plt.close()
    # python 一个折线图绘制多个曲线

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
#    plt.savefig('images/{0}.png'.format(name1), format='png')
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




    value3 = data3.values
    value4 = data4.values
    value1_bottom3=np.zeros_like(data1.T)
    value1_bottom5 = np.zeros_like(data1.T)
    value1_bottom10 = np.zeros_like(data1.T)
    value1_rank = np.zeros_like(data1.T)
    value1_bottom3 = np.zeros_like(data1.T)
    value1_bottom5 = np.zeros_like(data1.T)
    value1_bottom10 = np.zeros_like(data1.T)

    hs300_bottom3 = np.zeros_like(data1.T)
    hs300_bottom5 = np.zeros_like(data1.T)
    hs300_bottom10 = np.zeros_like(data1.T)
    hs300_bottom3 = np.zeros_like(data1.T)
    hs300_bottom5 = np.zeros_like(data1.T)
    hs300_bottom10 = np.zeros_like(data1.T)

    for num, fundid in enumerate(data1.columns[0:]):
        print("正在读取第{0}个月数据({1})".format(num+1 , fundid))

        value1 = data1[fundid].to_numpy()

        value11=np.argsort(value1)
        value13 = np.zeros_like(value11)
        for i in range(len(value11)):
            value12 = value11[i]
            value13[value12] = i

        value11 = value13

        value1_istop3 = value11 <=2
        value1_bottom3[num] = value1_istop3
        value1_istop5 = value11 <=4
        value1_bottom5[num] = value1_istop5
        value1_istop10 = value11 <=9
        value1_bottom10[num] = value1_istop10

        top3_hs300 = value1 <= value3[num]
        hs300_bottom3[num] = top3_hs300
        top5_hs300 = value1 <= value3[num]
        hs300_bottom5[num] = top5_hs300
        top10_hs300 = value1 <= value3[num]
        hs300_bottom10[num] = top10_hs300




    # <editor-fold desc="D策略月度排名及p值">
    value1 = data1.values.T

    value2 = data2.values

    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[1:] / value2[:-1]
    return_list2= value4[1:]/value4[:-1]

    '''以下是策略a11'''
    a1_bottom3=  np.multiply(return_list,value1_bottom3[:-1,:])
    a1_bottom5 = np.multiply(return_list, value1_bottom5[:-1,:])
    a1_bottom10 = np.multiply(return_list, value1_bottom10[:-1,:])

    a1_return_list_bottom3=a1_bottom3.sum(axis=1)/3
    a1_return_list_bottom5 = a1_bottom5.sum(axis=1) / 5
    a1_return_list_bottom10 = a1_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list,a1_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list,a1_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list,a1_return_list_bottom10))


    a1_strategy1_value_bottom3 = 28* np.ones(len(a1_strategy_value_bottom3))-sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28* np.ones(len(a1_strategy_value_bottom3))-sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28* np.ones(len(a1_strategy_value_bottom3))-sort(a1_strategy_value_bottom10)[:,-1]


    '''策略a11结束'''
    plot2(a1_strategy1_value_bottom3,'策略c1_bottom3_1个月_排名',date1,1,'排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c1_bottom5_1个月_排名', date1,1,'排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c1_bottom10_1个月_排名', date1,1,'排名情况')

    pd_b11_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c1_bottom3_1个月_排名'], index=date1[:-1])
    pd_b11_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c1_bottom5_1个月_排名'], index=date1[:-1])
    pd_b11_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c1_bottom10_1个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1/27 * np.ones(len(a1_strategy_value_bottom3))* sort(a1_strategy_value_bottom3)[:,-1]*100
    a1_strategy2_value_bottom5 = 1/27 * np.ones(len(a1_strategy_value_bottom3))*sort(a1_strategy_value_bottom5)[:,-1]*100
    a1_strategy2_value_bottom10 = 1/27 * np.ones(len(a1_strategy_value_bottom3))*sort(a1_strategy_value_bottom10)[:,-1]*100


    '''策略a12结束'''
    plot2(a1_strategy2_value_bottom3, '策略c1_bottom3_1个月_p值', date1, 1,'p值')
    plot2(a1_strategy2_value_bottom5, '策略c1_bottom5_1个月_p值', date1, 1,'p值')
    plot2(a1_strategy2_value_bottom10, '策略c1_bottom10_1个月_p值', date1, 1,'p值')

    pd_b12_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c1_bottom3_1个月_p值'], index=date1[:-1])
    pd_b12_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c1_bottom5_1个月_p值'], index=date1[:-1])
    pd_b12_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c1_bottom10_1个月_p值'], index=date1[:-1])



    date1=date111[:-2]
    value1 = data1.values.T

    value2 = data2.values

    value2 = value2 / value2[0]
    value4 = value4 / value4[0]
    return_list = value2[3:] / value2[:-3]
    return_list2 = value4[3:] / value4[:-3]

    '''以下是策略a11'''
    a1_bottom3 = np.multiply(return_list, value1_bottom3[:-3, :])
    a1_bottom5 = np.multiply(return_list, value1_bottom5[:-3, :])
    a1_bottom10 = np.multiply(return_list, value1_bottom10[:-3, :])

    a1_return_list_bottom3 = a1_bottom3.sum(axis=1) / 3
    a1_return_list_bottom5 = a1_bottom5.sum(axis=1) / 5
    a1_return_list_bottom10 = a1_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list, a1_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list, a1_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list, a1_return_list_bottom10))

    a1_strategy1_value_bottom3 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom10)[:,-1]

    '''策略a11结束'''
    plot2(a1_strategy1_value_bottom3, '策略c1_bottom3_3个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c1_bottom5_3个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c1_bottom10_3个月_排名', date1, 1, '排名情况')

    pd_b21_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c1_bottom3_3个月_排名'], index=date1[:-1])
    pd_b21_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c1_bottom5_3个月_排名'], index=date1[:-1])
    pd_b21_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c1_bottom10_3个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom3)[:,-1] * 100
    a1_strategy2_value_bottom5 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom5)[:,-1] * 100
    a1_strategy2_value_bottom10 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom10)[:,-1] * 100

    '''策略a12结束'''
    plot2(a1_strategy2_value_bottom3, '策略c1_bottom3_3个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom5, '策略c1_bottom5_3个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom10, '策略c1_bottom10_3个月_p值', date1, 1, 'p值')

    pd_b22_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c1_bottom3_3个月_p值'], index=date1[:-1])
    pd_b22_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c1_bottom5_3个月_p值'], index=date1[:-1])
    pd_b22_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c1_bottom10_3个月_p值'], index=date1[:-1])

    date1 = date111[:-5]
    value1 = data1.values.T

    value2 = data2.values

    value2 = value2 / value2[0]
    value4 = value4 / value4[0]
    return_list = value2[6:] / value2[:-6]
    return_list2 = value4[6:] / value4[:-6]

    '''以下是策略a11'''
    a1_bottom3 = np.multiply(return_list, value1_bottom3[:-6, :])
    a1_bottom5 = np.multiply(return_list, value1_bottom5[:-6, :])
    a1_bottom10 = np.multiply(return_list, value1_bottom10[:-6, :])

    a1_return_list_bottom3 = a1_bottom3.sum(axis=1) / 3
    a1_return_list_bottom5 = a1_bottom5.sum(axis=1) / 5
    a1_return_list_bottom10 = a1_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list, a1_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list, a1_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list, a1_return_list_bottom10))

    a1_strategy1_value_bottom3 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom10)[:,-1]

    '''策略a11结束'''
    plot2(a1_strategy1_value_bottom3, '策略c1_bottom3_6个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c1_bottom5_6个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c1_bottom10_6个月_排名', date1, 1, '排名情况')

    pd_b31_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c1_bottom3_6个月_排名'], index=date1[:-1])
    pd_b31_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c1_bottom5_6个月_排名'], index=date1[:-1])
    pd_b31_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c1_bottom10_6个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom3)[:,-1] * 100
    a1_strategy2_value_bottom5 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom5)[:,-1] * 100
    a1_strategy2_value_bottom10 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom10)[:,-1] * 100

    '''策略a12结束'''
    plot2(a1_strategy2_value_bottom3, '策略c1_bottom3_6个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom5, '策略c1_bottom5_6个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom10, '策略c1_bottom10_6个月_p值', date1, 1, 'p值')

    pd_b32_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c1_bottom3_6个月_p值'], index=date1[:-1])
    pd_b32_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c1_bottom5_6个月_p值'], index=date1[:-1])
    pd_b32_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c1_bottom10_6个月_p值'], index=date1[:-1])

    date1=date111
    value1 = data1.values.T

    value2 = data2.values

    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[1:] / value2[:-1]
    return_list2=value4[1:]/value4[:-1]
    '''以下是策略a2'''
    a2_hs300_bottom3 = np.multiply(hs300_bottom3[:-1, :], value1_bottom3[:-1, :])
    a2_hs300_bottom5 = np.multiply(hs300_bottom5[:-1, :], value1_bottom5[:-1, :])
    a2_hs300_bottom10 = np.multiply(hs300_bottom10[:-1, :], value1_bottom10[:-1, :])

    sum_hs300_bottom3=a2_hs300_bottom3.sum(axis=1)
    hs300_3x = 3* np.ones(len(sum_hs300_bottom3))
    hs300_bottom3_result=hs300_3x-sum_hs300_bottom3

    sum_hs300_bottom5 = a2_hs300_bottom5.sum(axis=1)
    hs300_5x = 5 * np.ones(len(sum_hs300_bottom5))
    hs300_bottom5_result = hs300_5x - sum_hs300_bottom5

    sum_hs300_bottom10 = a2_hs300_bottom10.sum(axis=1)
    hs300_10x = 10 * np.ones(len(sum_hs300_bottom10))
    hs300_bottom10_result = hs300_10x - sum_hs300_bottom10


    value_hs300_bottom3=np.column_stack((a2_hs300_bottom3,hs300_bottom3_result))
    value_hs300_bottom5 = np.column_stack((a2_hs300_bottom5, hs300_bottom5_result))
    value_hs300_bottom10 = np.column_stack((a2_hs300_bottom10, hs300_bottom10_result))

    return_list1=np.hstack((return_list,return_list2))

    a2_bottom3 = np.multiply(return_list1, value_hs300_bottom3)
    a2_bottom5 = np.multiply(return_list1, value_hs300_bottom5)
    a2_bottom10 = np.multiply(return_list1, value_hs300_bottom10)

    a2_return_list_bottom3 = a2_bottom3.sum(axis=1) / 3
    a2_return_list_bottom5 = a2_bottom5.sum(axis=1) / 5
    a2_return_list_bottom10 = a2_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list, a2_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list, a2_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list, a2_return_list_bottom10))

    a1_strategy1_value_bottom3 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom10)[:,-1]

    plot2(a1_strategy1_value_bottom3, '策略c2_bottom3_1个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c2_bottom5_1个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c2_bottom10_1个月_排名', date1, 1, '排名情况')

    pd_c11_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c2_bottom3_1个月_排名'], index=date1[:-1])
    pd_c11_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c2_bottom5_1个月_排名'], index=date1[:-1])
    pd_c11_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c2_bottom10_1个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom3)[:,-1] * 100
    a1_strategy2_value_bottom5 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom5)[:,-1] * 100
    a1_strategy2_value_bottom10 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom10)[:,-1] * 100


    plot2(a1_strategy2_value_bottom3, '策略c2_bottom3_1个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom5, '策略c2_bottom5_1个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom10, '策略c2_bottom10_1个月_p值', date1, 1, 'p值')

    pd_c12_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c2_bottom3_1个月_p值'], index=date1[:-1])
    pd_c12_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c2_bottom5_1个月_p值'], index=date1[:-1])
    pd_c12_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c2_bottom10_1个月_p值'], index=date1[:-1])

    '''策略a2结束'''

    date1=date111[:-2]
    value1 = data1.values.T

    value2 = data2.values

    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[3:] / value2[:-3]
    return_list2=value4[3:]/value4[:-3]
    '''以下是策略a21'''
    a2_hs300_bottom3 = np.multiply(hs300_bottom3[:-3, :], value1_bottom3[:-3, :])
    a2_hs300_bottom5 = np.multiply(hs300_bottom5[:-3, :], value1_bottom5[:-3, :])
    a2_hs300_bottom10 = np.multiply(hs300_bottom10[:-3, :], value1_bottom10[:-3, :])

    sum_hs300_bottom3=a2_hs300_bottom3.sum(axis=1)
    hs300_3x = 3* np.ones(len(sum_hs300_bottom3))
    hs300_bottom3_result=hs300_3x-sum_hs300_bottom3

    sum_hs300_bottom5 = a2_hs300_bottom5.sum(axis=1)
    hs300_5x = 5 * np.ones(len(sum_hs300_bottom5))
    hs300_bottom5_result = hs300_5x - sum_hs300_bottom5

    sum_hs300_bottom10 = a2_hs300_bottom10.sum(axis=1)
    hs300_10x = 10 * np.ones(len(sum_hs300_bottom10))
    hs300_bottom10_result = hs300_10x - sum_hs300_bottom10


    value_hs300_bottom3=np.column_stack((a2_hs300_bottom3,hs300_bottom3_result))
    value_hs300_bottom5 = np.column_stack((a2_hs300_bottom5, hs300_bottom5_result))
    value_hs300_bottom10 = np.column_stack((a2_hs300_bottom10, hs300_bottom10_result))

    return_list1=np.hstack((return_list,return_list2))

    a2_bottom3 = np.multiply(return_list1, value_hs300_bottom3)
    a2_bottom5 = np.multiply(return_list1, value_hs300_bottom5)
    a2_bottom10 = np.multiply(return_list1, value_hs300_bottom10)

    a2_return_list_bottom3 = a2_bottom3.sum(axis=1) / 3
    a2_return_list_bottom5 = a2_bottom5.sum(axis=1) / 5
    a2_return_list_bottom10 = a2_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list, a2_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list, a2_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list, a2_return_list_bottom10))
    a1_strategy1_value_bottom3 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom10)[:,-1]

    plot2(a1_strategy1_value_bottom3, '策略c2_bottom3_3个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c2_bottom5_3个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c2_bottom10_3个月_排名', date1, 1, '排名情况')
    pd_c21_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c2_bottom3_3个月_排名'], index=date1[:-1])
    pd_c21_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c2_bottom5_3个月_排名'], index=date1[:-1])
    pd_c21_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c2_bottom10_3个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom3)[:,-1] * 100
    a1_strategy2_value_bottom5 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom5)[:,-1] * 100
    a1_strategy2_value_bottom10 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom10)[:,-1] * 100

    plot2(a1_strategy2_value_bottom3, '策略c2_bottom3_3个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom5, '策略c2_bottom5_3个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom10, '策略c2_bottom10_3个月_p值', date1, 1, 'p值')
    pd_c22_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c2_bottom3_3个月_p值'], index=date1[:-1])
    pd_c22_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c2_bottom5_3个月_p值'], index=date1[:-1])
    pd_c22_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c2_bottom10_3个月_p值'], index=date1[:-1])

    '''策略a21结束'''

    date1=date111[:-5]
    value1 = data1.values.T

    value2 = data2.values

    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[6:] / value2[:-6]
    return_list2=value4[6:]/value4[:-6]
    '''以下是策略a2'''
    a2_hs300_bottom3 = np.multiply(hs300_bottom3[:-6, :], value1_bottom3[:-6, :])
    a2_hs300_bottom5 = np.multiply(hs300_bottom5[:-6, :], value1_bottom5[:-6, :])
    a2_hs300_bottom10 = np.multiply(hs300_bottom10[:-6, :], value1_bottom10[:-6, :])

    sum_hs300_bottom3=a2_hs300_bottom3.sum(axis=1)
    hs300_3x = 3* np.ones(len(sum_hs300_bottom3))
    hs300_bottom3_result=hs300_3x-sum_hs300_bottom3

    sum_hs300_bottom5 = a2_hs300_bottom5.sum(axis=1)
    hs300_5x = 5 * np.ones(len(sum_hs300_bottom5))
    hs300_bottom5_result = hs300_5x - sum_hs300_bottom5

    sum_hs300_bottom10 = a2_hs300_bottom10.sum(axis=1)
    hs300_10x = 10 * np.ones(len(sum_hs300_bottom10))
    hs300_bottom10_result = hs300_10x - sum_hs300_bottom10


    value_hs300_bottom3=np.column_stack((a2_hs300_bottom3,hs300_bottom3_result))
    value_hs300_bottom5 = np.column_stack((a2_hs300_bottom5, hs300_bottom5_result))
    value_hs300_bottom10 = np.column_stack((a2_hs300_bottom10, hs300_bottom10_result))

    return_list1=np.hstack((return_list,return_list2))

    a2_bottom3 = np.multiply(return_list1, value_hs300_bottom3)
    a2_bottom5 = np.multiply(return_list1, value_hs300_bottom5)
    a2_bottom10 = np.multiply(return_list1, value_hs300_bottom10)

    a2_return_list_bottom3 = a2_bottom3.sum(axis=1) / 3
    a2_return_list_bottom5 = a2_bottom5.sum(axis=1) / 5
    a2_return_list_bottom10 = a2_bottom10.sum(axis=1) / 10

    a1_strategy_value_bottom3 = np.column_stack((return_list, a2_return_list_bottom3))
    a1_strategy_value_bottom5 = np.column_stack((return_list, a2_return_list_bottom5))
    a1_strategy_value_bottom10 = np.column_stack((return_list, a2_return_list_bottom10))
    a1_strategy1_value_bottom3 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom3)[:,-1]
    a1_strategy1_value_bottom5 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom5)[:,-1]
    a1_strategy1_value_bottom10 = 28 * np.ones(len(a1_strategy_value_bottom3)) - sort(a1_strategy_value_bottom10)[:,-1]

    '''策略a11结束'''
    plot2(a1_strategy1_value_bottom3, '策略c2_bottom3_6个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom5, '策略c2_bottom5_6个月_排名', date1, 1, '排名情况')
    plot2(a1_strategy1_value_bottom10, '策略c2_bottom10_6个月_排名', date1, 1, '排名情况')

    pd_c31_bottom3 = pd.DataFrame(a1_strategy1_value_bottom3, columns=['策略c2_bottom3_6个月_排名'], index=date1[:-1])
    pd_c31_bottom5 = pd.DataFrame(a1_strategy1_value_bottom5, columns=['策略c2_bottom5_6个月_排名'], index=date1[:-1])
    pd_c31_bottom10 = pd.DataFrame(a1_strategy1_value_bottom10, columns=['策略c2_bottom10_6个月_排名'], index=date1[:-1])

    a1_strategy2_value_bottom3 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom3)[:,-1] * 100
    a1_strategy2_value_bottom5 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom5)[:,-1] * 100
    a1_strategy2_value_bottom10 = 1 / 27 * np.ones(len(a1_strategy_value_bottom3)) * sort(a1_strategy_value_bottom10)[:,-1] * 100


    plot2(a1_strategy2_value_bottom3, '策略c2_bottom3_6个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom5, '策略c2_bottom5_6个月_p值', date1, 1, 'p值')
    plot2(a1_strategy2_value_bottom10, '策略c2_bottom10_6个月_p值', date1, 1, 'p值')
    pd_c32_bottom3 = pd.DataFrame(a1_strategy2_value_bottom3, columns=['策略c2_bottom3_6个月_p值'], index=date1[:-1])
    pd_c32_bottom5 = pd.DataFrame(a1_strategy2_value_bottom5, columns=['策略c2_bottom5_6个月_p值'], index=date1[:-1])
    pd_c32_bottom10 = pd.DataFrame(a1_strategy2_value_bottom10, columns=['策略c2_bottom10_6个月_p值'], index=date1[:-1])
    # </editor-fold>


    pd.concat([pd_b11_bottom3, pd_b21_bottom3, pd_b31_bottom3,pd_b11_bottom5,pd_b21_bottom5, pd_b31_bottom5,pd_b11_bottom10, pd_b21_bottom10,pd_b31_bottom10, pd_c11_bottom3, pd_c21_bottom3,pd_c31_bottom3,pd_c11_bottom5,pd_c21_bottom5, pd_c31_bottom5,pd_c11_bottom10,pd_c21_bottom10,pd_c31_bottom10], axis=1).to_excel(
        r'C策略/策略排名及p值/投资组合月度排名_bottom.xlsx', encoding="gbk")
    pd.concat([pd_b12_bottom3, pd_b22_bottom3, pd_b32_bottom3,pd_b12_bottom5,pd_b22_bottom5,  pd_b32_bottom5,pd_b12_bottom10,pd_b22_bottom10,pd_b32_bottom10, pd_c12_bottom3, pd_c22_bottom3,pd_c32_bottom3,pd_c12_bottom5,pd_c22_bottom5, pd_c32_bottom5,pd_c12_bottom10,pd_c22_bottom10,pd_c32_bottom10], axis=1).to_excel(
        r'C策略/策略排名及p值/投资组合月度p值_bottom.xlsx', encoding="gbk")
    pd.read_excel(r'C策略/策略排名及p值/投资组合月度排名_bottom.xlsx', index_col=0,skiprows=lambda x: x > 0 and (x - 1) % 3 != 0).to_excel(
        r'C策略/策略排名及p值/投资组合季度排名_bottom.xlsx', encoding="gbk")
    pd.read_excel(r'C策略/策略排名及p值/投资组合月度p值_bottom.xlsx', index_col=0,skiprows=lambda x: x > 0 and (x - 1) % 3 != 0).to_excel(
        r'C策略/策略排名及p值/投资组合季度p值_bottom.xlsx', encoding="gbk")







