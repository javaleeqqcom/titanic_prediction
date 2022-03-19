# -*- coding: utf-8 -*-
# @Time    : 2022/3/18 15:14
# @Author  : Jiawei_Lee
# @Email   : java_lee@qq.com
# @File    : preparation.py
# @Software: PyCharm

from collections import Counter
import pandas as pd
import re

nan_i=[]

def getCategoryFrequency(data:pd.Series,maximun:int=0,prefix:str=""):
    '''getCategoryFrequency(data [,maximun=0,prefix=""])
    根据输入序列，将Category类型属性处理成频率矩阵，矩阵的每一列表示其值为该行该类型的概率，对于非缺失值为哪个值则哪一列为1.0其余列均为0.0。
    缺失值按照所有类型的频率均摊，也只有缺失值所在行才会出现在区间(0.0,1.0)的元素。
        data: pandas序列，其值通常为字符串或整型，可以有缺失值。
        maximun: 最终输出结果的列数上限，为0则不设上限。当data中元素的种类数（不含缺失值）超过maximun时，则取频次较少的类型合并为'others'类型，使得输出列数恰为maximun。
    '''
    bool_isna=data.isna()
    ele_count=Counter(data.loc[~bool_isna])   # 缺失值不会统计在内
    print(ele_count)
    totol=sum(ele_count.values())
    if 0==totol:return None
    ele_count=sorted(list(ele_count.items()),key=lambda e:(-e[1],str(e[0])))      # 按照频率降序排序
    if maximun>0 and len(ele_count)>maximun:
        ele_count=ele_count[:maximun-1]+('others',sum(e[1] for e in ele_count[maximun-1:]))
    FrequencyDict={e[0]:[0.0,]*i+[1.0,]+[0.0]*(len(ele_count)-i-1) for i,e in enumerate(ele_count)}
    nan_Frequency=[e[1]/totol for e in ele_count]
    matrix = [None] * len(data)
    for i in range(len(data)):
        if bool_isna[i]:
            matrix[i]=nan_Frequency
            nan_i.append(i)
        else:
            matrix[i]=FrequencyDict[str(data.loc[i])]
    return pd.DataFrame(matrix,columns=[prefix+e[0] for e in ele_count],dtype='float')

def words_statistic(string_array:pd.Series,pattern:str):
    '''
    :param string_array: 设string_array=[s1,s2,...,sn]，si表示一个句子，每一个句子有0到多个单词。
    :param pattern: 匹配单词的正则表达式，不是分隔符的正则表达式！
    :return: 返回以['key','count']为列的DataFrame，其中每一行对应一种“单词”，其'count'值为最多在多少个句子中出现过，注意不是在所有句子中出现的总次数（同一个句子中出现多次某个单词，此单词也只计一次），所以'count'不可能超过n。
    统计字符串序列 string_array 中的特征"单词"，“单词”不一定是标准的英文单词格式，而是按照正则表达式 pattern 来匹配。
    '''
    pattern=re.compile(pattern)
    words=Counter(word for sentence in string_array for word in set(re.findall(pattern,sentence)))
    words=list(words.items())
    words.sort(key=lambda k_c:(-k_c[1],k_c[0])) # 按频次降序排序
    return pd.DataFrame(words,columns=['key','count'])

def get_title(string_list:pd.Series,key_list:pd.Series):
    '''
    :param string_list: string_list.shape=(n,)
    :param key_list: key_list.shape=(m,)
    :return:title_matrix.shape=(n,m)
    从字符串string_list[i]特征中提取关键字符串key，并以所有关键字符串为列构造一个布尔矩阵title_matrix（以整型存储）。title_matrix[i][k]表示i号字符串中是否包含key_list[k]。
    '''
    title_matrix=[None]*len(key_list)
    for i,key in enumerate(key_list):
        title_matrix[i]=list(int(key in name) for name in string_list)
    title_matrix=pd.concat(tuple(map(pd.Series,title_matrix)),axis=1)
    title_matrix.columns=key_list
    return title_matrix

train = pd.read_csv("train.csv")
ret=words_statistic(train['Name'],"\w+")
print(ret)