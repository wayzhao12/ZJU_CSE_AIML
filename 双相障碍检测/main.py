#如有报错，请根据系统测试报错结果修改

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from minepy import MINE


def data_processing_and_feature_selecting(data_path):
    """
    特征选择
    :param  data_path: 数据集路径
    :return: new_features,label: 经过预处理和特征选择后的特征数据、类标数据
    """ 
    
    #数据路径
    # data_path = "DataSet.xlsx"
    #导入医疗数据
    data_xls = pd.ExcelFile(data_path)
    data={}
    
    #查看数据名称与大小
    for name in data_xls.sheet_names:
            df = data_xls.parse(sheet_name=name,header=None)
            data[name] = df
    
    #获取 特征1 特征2 类标    
    feature1_raw = data['Feature1']
    feature2_raw = data['Feature2']
    label = data['label']


    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    feature1 = pd.DataFrame(scaler.fit_transform(feature1_raw))
    feature2 = pd.DataFrame(scaler.fit_transform(feature2_raw))

    features = pd.concat([feature1, feature2], axis=1)

    # 统计特征值和label的皮尔孙相关系数  进行排序筛选特征
    select_feature_number = 10
    select_features = SelectKBest(lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T)),
                                  k=select_feature_number).fit(features,np.array(label).flatten()).get_support(indices=True)

    # 查看提取的特征序号
    # print("查看提取的特征序号:", select_features)

    # 特征选择
    new_features = features[features.columns.values[select_features]]

    # 返回筛选后的数据
    return new_features,label

# 加载模型(请加载你认为的最佳模型)
model_path = 'results/my_model.m'

# 加载模型
model = joblib.load(model_path)

# ---------------------------------------------------------------------------

def predict(new_features):
    """
    加载模型和模型预测
    :param  new_features : 测试数据
    :return y_predict : 预测结果
    """
    y_predict = model.predict(new_features)
    
    return y_predict
