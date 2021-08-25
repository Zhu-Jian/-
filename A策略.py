import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
import os

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
#    plt.savefig('A策略/持仓明细/{0}-{1}.png'.format(name1,name2), format='png')
    plt.close()
    # python 一个折线图绘制多个曲线


#以下是月度数据处理函数


if __name__ == '__main__':

    # 载入数据
    data1 = pd.read_excel('基础数据/申万指数得分.xlsx',header=0,index_col=0).T
    data2 = pd.read_excel('基础数据/申万指数收盘价.xlsx',header=0,index_col=0)
    data3 = pd.read_excel('基础数据/沪深300得分.xlsx',header=0,index_col=0)
    data4 = pd.read_excel('基础数据/沪深300收盘价.xlsx',header=0,index_col=0)

    date1=data1.columns.values
    data_index=data1.T
    date_index=data_index.columns.values
    date_index2=np.append(date_index,'SH000300')
    date2=date1[np.linspace(0, 138, 47).astype(int)]



    value3 = data3.values
    value4 = data4.values
    # <editor-fold desc="初始化矩阵，并判断行业得分排名（得分最高排名27，得分最低排名0）">

    value1_top3=np.zeros_like(data1.T)
    value1_top5 = np.zeros_like(data1.T)
    value1_top10 = np.zeros_like(data1.T)
    value1_rank = np.zeros_like(data1.T)
    value1_bottom3 = np.zeros_like(data1.T)
    value1_bottom5 = np.zeros_like(data1.T)
    value1_bottom10 = np.zeros_like(data1.T)

    hs300_top3 = np.zeros_like(data1.T)
    hs300_top5 = np.zeros_like(data1.T)
    hs300_top10 = np.zeros_like(data1.T)
    hs300_bottom3 = np.zeros_like(data1.T)
    hs300_bottom5 = np.zeros_like(data1.T)
    hs300_bottom10 = np.zeros_like(data1.T)




    for num, fundid in enumerate(data1.columns[0:]):
        print("正在读取第{0}个月数据({1})".format(num+1 , fundid))

        value1 = data1[fundid].to_numpy()
        value11 = np.argsort(value1)
        value13=np.zeros_like(value11)
        for i in range(len(value11)):
            value12=value11[i]
            value13[value12]=i

        value11=value13

        value1_istop3 = value11 >= 25
        value1_top3[num] = value1_istop3
        value1_istop5 = value11 >= 23
        value1_top5[num] = value1_istop5
        value1_istop10 = value11 >= 18
        value1_top10[num] = value1_istop10

        top3_hs300 = value1 >= value3[num]
        hs300_top3[num] = top3_hs300
        top5_hs300 = value1 >= value3[num]
        hs300_top5[num] = top5_hs300
        top10_hs300 = value1 >= value3[num]
        hs300_top10[num] = top10_hs300

    # </editor-fold>


    value1 = data1.values.T

    value2 = data2.values

    value2=value2/value2[0]
    value4=value4/value4[0]
    return_list = value2[1:] / value2[:-1]
    return_list2=value4[1:]/value4[:-1]

    # <editor-fold desc="策略a1">
    '''以下是策略a1'''
    #提取策略持仓股的收益情况
    a1_top3=  np.multiply(return_list,value1_top3[:-1,:])
    a1_top5 = np.multiply(return_list, value1_top5[:-1,:])
    a1_top10 = np.multiply(return_list, value1_top10[:-1,:])
    #计算组合净值变动比例
    a1_return_list_top3=a1_top3.sum(axis=1)/3
    a1_return_list_top5 = a1_top5.sum(axis=1) / 5
    a1_return_list_top10 = a1_top10.sum(axis=1) / 10
    #计算净值数据
    a1_strategy_value_top3 = np.cumprod(a1_return_list_top3)
    a1_strategy_value_top5 = np.cumprod(a1_return_list_top5)
    a1_strategy_value_top10 = np.cumprod(a1_return_list_top10)
    hs300_value=np.cumprod(return_list2)
    #调整净值数据（设定起始日净值为1）
    a1_strategy_value_top3 = np.insert(a1_strategy_value_top3,0,1)
    a1_strategy_value_top5 = np.insert(a1_strategy_value_top5,0,1)
    a1_strategy_value_top10 = np.insert(a1_strategy_value_top10,0,1)
    hs300_value = np.insert(hs300_value,0,1)
    #绘制策略净值曲线
    plot(a1_strategy_value_top3,hs300_value,'策略a1_top3','沪深300',date1)
    plot(a1_strategy_value_top5, hs300_value, '策略a1_top5', '沪深300', date1)
    plot(a1_strategy_value_top10, hs300_value, '策略a1_top10', '沪深300', date1)
    #保存数据
    pd_a1_top3=pd.DataFrame(a1_strategy_value_top3,columns=['策略a1_top3'],index=date1)
    pd_a1_top5 = pd.DataFrame(a1_strategy_value_top5, columns=['策略a1_top5'], index=date1)
    pd_a1_top10 = pd.DataFrame(a1_strategy_value_top10, columns=['策略a1_top10'], index=date1)
    '''策略a1结束'''
    # </editor-fold>

    # <editor-fold desc="策略a2">
    '''以下是策略a2'''
    a2_hs300_top3 = np.multiply(hs300_top3[:-1, :], value1_top3[:-1, :])
    a2_hs300_top5 = np.multiply(hs300_top5[:-1, :], value1_top5[:-1, :])
    a2_hs300_top10 = np.multiply(hs300_top10[:-1, :], value1_top10[:-1, :])

    sum_hs300_top3=a2_hs300_top3.sum(axis=1)
    hs300_3x = 3* np.ones(len(sum_hs300_top3))
    hs300_top3_result=hs300_3x-sum_hs300_top3

    sum_hs300_top5 = a2_hs300_top5.sum(axis=1)
    hs300_5x = 5 * np.ones(len(sum_hs300_top5))
    hs300_top5_result = hs300_5x - sum_hs300_top5

    sum_hs300_top10 = a2_hs300_top10.sum(axis=1)
    hs300_10x = 10 * np.ones(len(sum_hs300_top10))
    hs300_top10_result = hs300_10x - sum_hs300_top10


    value_hs300_top3 =np.column_stack((a2_hs300_top3,hs300_top3_result))
    value_hs300_top5 = np.column_stack((a2_hs300_top5, hs300_top5_result))
    value_hs300_top10 = np.column_stack((a2_hs300_top10, hs300_top10_result))

    return_list1=np.hstack((return_list,return_list2))

    a2_top3 = np.multiply(return_list1, value_hs300_top3)
    a2_top5 = np.multiply(return_list1, value_hs300_top5)
    a2_top10 = np.multiply(return_list1, value_hs300_top10)

    a2_return_list_top3 = a2_top3.sum(axis=1) / 3
    a2_return_list_top5 = a2_top5.sum(axis=1) / 5
    a2_return_list_top10 = a2_top10.sum(axis=1) / 10

    a2_strategy_value_top3 = np.cumprod(a2_return_list_top3)
    a2_strategy_value_top5 = np.cumprod(a2_return_list_top5)
    a2_strategy_value_top10 = np.cumprod(a2_return_list_top10)
    hs300_value = np.cumprod(return_list2)
    a2_strategy_value_top3 = np.insert(a2_strategy_value_top3, 0, 1)
    a2_strategy_value_top5 = np.insert(a2_strategy_value_top5, 0, 1)
    a2_strategy_value_top10 = np.insert(a2_strategy_value_top10, 0, 1)
    hs300_value = np.insert(hs300_value, 0, 1)


    plot(a2_strategy_value_top3, hs300_value, '策略a2_top3', '沪深300', date1)
    plot(a2_strategy_value_top5, hs300_value, '策略a2_top5', '沪深300', date1)
    plot(a2_strategy_value_top10, hs300_value, '策略a2_top10', '沪深300', date1)
    pd_a2_top3 = pd.DataFrame(a2_strategy_value_top3, columns=['策略a2_top3'], index=date1)
    pd_a2_top5 = pd.DataFrame(a2_strategy_value_top5, columns=['策略a2_top5'], index=date1)
    pd_a2_top10 = pd.DataFrame(a2_strategy_value_top10, columns=['策略a2_top10'], index=date1)

    '''策略a2结束'''
    # </editor-fold>

    #保存持仓明细
    pd.DataFrame(value1_top3, columns=date_index, index=date1).to_excel(r"A策略/持仓明细/a1_top3_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value1_top5, columns=date_index, index=date1).to_excel(r"A策略/持仓明细/a1_top5_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value1_top10, columns=date_index, index=date1).to_excel(r"A策略/持仓明细/a1_top10_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value_hs300_top3, columns=date_index2, index=date1[:-1]).to_excel(r"A策略/持仓明细/a2_top3_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value_hs300_top5, columns=date_index2, index=date1[:-1]).to_excel(r"A策略/持仓明细/a2_top5_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value_hs300_top10, columns=date_index2, index=date1[:-1]).to_excel(r"A策略/持仓明细/a2_top10_持仓明细.xlsx", encoding="gbk")

    #设置季度数据
    value2 = data2.values
    value2 = value2[np.linspace(0, 138, 47).astype(int)]
    value2 = value2 / value2[0]
    value4 = value4[np.linspace(0, 138, 47).astype(int)]
    value4 = value4 / value4[0]
    return_list = value2[1:] / value2[:-1]
    return_list2 = value4[1:] / value4[:-1]

    # <editor-fold desc="策略a3">
    '''以下是策略a3'''
    value1_top3=value1_top3[np.linspace(0, 138, 47).astype(int)]
    value1_top5=value1_top5[np.linspace(0, 138, 47).astype(int)]
    value1_top10=value1_top10[np.linspace(0, 138, 47).astype(int)]
    a1_top3 = np.multiply(return_list, value1_top3[:-1, :])
    a1_top5 = np.multiply(return_list, value1_top5[:-1, :])
    a1_top10 = np.multiply(return_list, value1_top10[:-1, :])

    a1_return_list_top3 = a1_top3.sum(axis=1) / 3
    a1_return_list_top5 = a1_top5.sum(axis=1) / 5
    a1_return_list_top10 = a1_top10.sum(axis=1) / 10

    a1_strategy_value_top3 = np.cumprod(a1_return_list_top3)
    a1_strategy_value_top5 = np.cumprod(a1_return_list_top5)
    a1_strategy_value_top10 = np.cumprod(a1_return_list_top10)
    hs300_value = np.cumprod(return_list2)
    a1_strategy_value_top3 = np.insert(a1_strategy_value_top3,0,1)
    a1_strategy_value_top5 = np.insert(a1_strategy_value_top5,0,1)
    a1_strategy_value_top10 = np.insert(a1_strategy_value_top10,0,1)
    hs300_value = np.insert(hs300_value,0,1)
    plot(a1_strategy_value_top3, hs300_value, '策略a3_top3', '沪深300', date2)
    plot(a1_strategy_value_top5, hs300_value, '策略a3_top5', '沪深300', date2)
    plot(a1_strategy_value_top10, hs300_value, '策略a3_top10', '沪深300', date2)
    pd_a3_top3 = pd.DataFrame(a1_strategy_value_top3, columns=['策略a3_top3'], index=date2)
    pd_a3_top5 = pd.DataFrame(a1_strategy_value_top5, columns=['策略a3_top5'], index=date2)
    pd_a3_top10 = pd.DataFrame(a1_strategy_value_top10, columns=['策略a3_top10'], index=date2)
    '''策略a3结束'''
    # </editor-fold>
    pd.DataFrame(value1_top3, columns=date_index, index=date2).to_excel(r"A策略/持仓明细/a3_top3_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value1_top5, columns=date_index, index=date2).to_excel(r"A策略/持仓明细/a3_top5_持仓明细.xlsx", encoding="gbk")
    pd.DataFrame(value1_top10, columns=date_index, index=date2).to_excel(r"A策略/持仓明细/a3_top10_持仓明细.xlsx", encoding="gbk")

    # <editor-fold desc="策略a4">
    '''以下是策略a4'''
    hs300_top3=hs300_top3[np.linspace(0, 138, 47).astype(int)]
    hs300_top5=hs300_top5[np.linspace(0, 138, 47).astype(int)]
    hs300_top10=hs300_top10[np.linspace(0, 138, 47).astype(int)]
    a2_hs300_top3 = np.multiply(hs300_top3[:-1, :], value1_top3[:-1, :])
    a2_hs300_top5 = np.multiply(hs300_top5[:-1, :], value1_top5[:-1, :])
    a2_hs300_top10 = np.multiply(hs300_top10[:-1, :], value1_top10[:-1, :])

    sum_hs300_top3 = a2_hs300_top3.sum(axis=1)
    hs300_3x = 3 * np.ones(len(sum_hs300_top3))
    hs300_top3_result = hs300_3x - sum_hs300_top3

    sum_hs300_top5 = a2_hs300_top5.sum(axis=1)
    hs300_5x = 5 * np.ones(len(sum_hs300_top5))
    hs300_top5_result = hs300_5x - sum_hs300_top5

    sum_hs300_top10 = a2_hs300_top10.sum(axis=1)
    hs300_10x = 10 * np.ones(len(sum_hs300_top10))
    hs300_top10_result = hs300_10x - sum_hs300_top10

    value_hs300_top3 = np.column_stack((a2_hs300_top3, hs300_top3_result))
    value_hs300_top5 = np.column_stack((a2_hs300_top5, hs300_top5_result))
    value_hs300_top10 = np.column_stack((a2_hs300_top10, hs300_top10_result))

    return_list1 = np.hstack((return_list, return_list2))

    a2_top3 = np.multiply(return_list1, value_hs300_top3)
    a2_top5 = np.multiply(return_list1, value_hs300_top5)
    a2_top10 = np.multiply(return_list1, value_hs300_top10)

    a2_return_list_top3 = a2_top3.sum(axis=1) / 3
    a2_return_list_top5 = a2_top5.sum(axis=1) / 5
    a2_return_list_top10 = a2_top10.sum(axis=1) / 10

    a2_strategy_value_top3 = np.cumprod(a2_return_list_top3)
    a2_strategy_value_top5 = np.cumprod(a2_return_list_top5)
    a2_strategy_value_top10 = np.cumprod(a2_return_list_top10)
    hs300_value = np.cumprod(return_list2)
    a2_strategy_value_top3 = np.insert(a2_strategy_value_top3, 0, 1)
    a2_strategy_value_top5 = np.insert(a2_strategy_value_top5, 0, 1)
    a2_strategy_value_top10 = np.insert(a2_strategy_value_top10, 0, 1)
    hs300_value = np.insert(hs300_value, 0, 1)

    plot(a2_strategy_value_top3, hs300_value, '策略a4_top3', '沪深300', date2)
    plot(a2_strategy_value_top5, hs300_value, '策略a4_top5', '沪深300', date2)
    plot(a2_strategy_value_top10, hs300_value, '策略a4_top10', '沪深300', date2)
    pd_a4_top3 = pd.DataFrame(a2_strategy_value_top3, columns=['策略a4_top3'], index=date2)
    pd_a4_top5 = pd.DataFrame(a2_strategy_value_top5, columns=['策略a4_top5'], index=date2)
    pd_a4_top10 = pd.DataFrame(a2_strategy_value_top10, columns=['策略a4_top10'], index=date2)

    '''策略a4结束'''
    # </editor-fold>

    pd.DataFrame(value_hs300_top3, columns=date_index2, index=date2[:-1]).to_excel(r"A策略/持仓明细/a4_top3_持仓明细.xlsx",
                                                                                   encoding="gbk")
    pd.DataFrame(value_hs300_top5, columns=date_index2, index=date2[:-1]).to_excel(r"A策略/持仓明细/a4_top5_持仓明细.xlsx",
                                                                                   encoding="gbk")
    pd.DataFrame(value_hs300_top10, columns=date_index2, index=date2[:-1]).to_excel(r"A策略/持仓明细/a4_top10_持仓明细.xlsx",
                                                                                    encoding="gbk")




    pd.concat([pd_a1_top3,pd_a1_top5,pd_a1_top10,pd_a2_top3,pd_a2_top5,pd_a2_top10],axis=1).to_excel(r'A策略/A策略月度调仓组合净值.xlsx', encoding="gbk")
    pd.concat([pd_a3_top3,pd_a3_top5,pd_a3_top10,pd_a4_top3,pd_a4_top5,pd_a4_top10],axis=1).to_excel(r'A策略/A策略季度调仓组合净值.xlsx', encoding="gbk")

    """  ========================================================================================================
    设置数据保存目录
    """
    PATH_DATA = os.path.abspath('./A策略/持仓明细')
    print(PATH_DATA)
    """  ========================================================================================================
    根据已有的数据文件判断需要更新的内容
    """
    data_files_cfets = pd.DataFrame(columns=['filename', 'filetype', 'date'])
    filelist = [file for file in os.listdir(PATH_DATA) if file.find("持仓明细") >= 0]
    data_files_cfets['filename'] = filelist
    data_files_cfets['filetype'] = data_files_cfets['filename'].str.extract(r'(\d+)')
    data_files_cfets['date'] = data_files_cfets['filename'].str.extract(r'(\d+)')
    data_files_cfets = data_files_cfets.dropna()

    for loop_date, loop_row in data_files_cfets.iterrows():
        hold_list = []
        print("正在读取{}文件".format(loop_row['filename']))
        t1 = pd.read_excel(f"{PATH_DATA}/{loop_row['filename']}")
        name = t1.columns[1:].values
        date = t1.values[:, 0]
        t2 = t1.values[:, 1:]
        date_value = []
        name_value = []

        for i in range(len(t2)):
            date_value2 = date[i]

            name_value1 = []
            for j in range(np.size(t2, 1)):

                if t2[i, j] > 0:
                    name_value2 = name[j]
                    name_value1.append(name_value2)
            name_value.append(name_value1)
            date_value.append(date_value2)

        pd.DataFrame(name_value, index=date_value).to_excel(
            r'A策略/{}'.format(loop_row['filename']))
