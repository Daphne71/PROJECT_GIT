import pymssql
import argparse

import DBtype
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from tensorflow.keras import layers
#추가
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation,LSTM
from datetime import datetime
from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import set_random_seed
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pylab as plb
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
# 한글 깨짐 현상 방지
plb.rcParams['font.family'] = 'Malgun Gothic'
plb.rcParams['font.size'] = 12

class Common_Code():
    
    def __init__(self):
        #self.time = str(datetime.now()) <- C#에서 Create_date로 넘어오기 때문
        self.MODEL_BASE_PATH = "./model/"
        self.DATA_BASE_PATH = "./data/"
        self.IMG_PATH = "./img/"
        self.ERROR = 'Fail0'
        
        # 데이터 insert할 DB 정보 설정
        self.HOST = DBtype.DLS002['HOST']
        self.USER = DBtype.DLS002['USER']
        self.PW = DBtype.DLS002['PW']
        self.DB = DBtype.DLS002['DB']

    def error_msg(self):
        return self.ERROR

    # Common 추가 사항
    #수정 부분 ----------------------------------------

    def random_forest_img(self, x_col, y_col, forest, algorithm_name, model_id):
        path = self.IMG_PATH + algorithm_name + "/"

        if os.path.isdir(path) is False:
            os.mkdir(path)
        try:
            imp = forest.feature_importances_
            imp_len = len(imp)
            colnames = list(x_col.split(","))

            plt.barh(range(imp_len), imp)
            plt.yticks(range(0, imp_len), (colnames))
            plt.title("종속변수" + y_col + "에 대한 독립변수의 중요도")
            plt.savefig(path+ model_id + '.png', format='png', dpi=300, bbox_inches='tight')
            plt.close()
            msg = "Success"
        except:
            msg = "Fail"
        return msg

    #path 경로만 잘 설정하면 될 듯
    def decision_tree_img(self, x_col, y_train, dct, model_id, algorithm_name="decision_tree"):
        path = self.IMG_PATH + algorithm_name + "/"

        if os.path.isdir(path) is False:
            os.mkdir(path)
        try:
            unique_array = np.unique(y_train)
            print("0")
            fig = plt.figure(figsize=(15, 8))
            _ = tree.plot_tree(dct,
                               feature_names=x_col,
                               class_names=[str(i) for i in unique_array],
                               filled=True)
            print("1")

            # 도식화 이미지 저장
            plt.savefig(path + model_id + '.png', format='png', dpi=300)
            print("2")
            plt.show()
            print("3")
            
            plt.close()
            msg = "Success"
        except:
            msg = "Fail"
            plt.savefig(path + model_id + '.png', format='png', dpi=300)
        return msg


#수정했으니 꼭 수정해야됨!
    def reshape_3(self, X_train, X_test):
        x_train_re = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
        x_test_re = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

        return x_train_re, x_test_re

    def lstm(self, X_train, n_classes):

        inputs = Input(shape=(X_train.shape[1], 1))

        x = LSTM(120, return_sequences=True)(inputs)
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32, return_sequences=True)(x)
        x = LSTM(16)(x)
        outputs = Dense(n_classes, activation='softmax')(x)

        lstm = Model(inputs=inputs, outputs=outputs)
        lstm.summary()

        lstm.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        return lstm

    # train : ds, y / pred_y = yhat1 : 예측 y 값 / result : 예측할 월의 ds, y , yhat
    def neuralProphet(self, dataset, train_data, x_col, y_col, ref_month, pred_month, pred_cycle):
        train_data = pd.DataFrame({"ds": pd.to_datetime(train_data[x_col]), 'y': train_data[y_col]})

        NP = NeuralProphet()
        if pred_cycle == "0":
            metrics = NP.fit(train_data, freq='MS')
        elif pred_cycle == "1":
            metrics = NP.fit(train_data, freq='D')

        # pred_month개월 후를 예측하는 데이터프레임
        future = NP.make_future_dataframe(train_data, periods=pred_month)
        y_predict = NP.predict(future)

        result = y_predict[['ds', 'yhat1']]
        result.loc[result['yhat1'] < 0, 'yhat1'] = 0
        if pred_cycle == "0":
            result['ds'] = result['ds'].astype(str).str[0:7]
        elif pred_cycle == "1":
            result['ds'] = result['ds'].astype(str).str[0:10]
        result = result.set_index('ds')
        data = dataset.set_index(x_col)

        result = pd.concat([result, data], axis=1)

        result = result.reset_index()

        if pred_cycle == "0":
            last_month = str(datetime.strptime(ref_month[0:7], '%Y-%m') + relativedelta(months=(pred_month - 1)))
        elif pred_cycle == "1":
            last_month = str(datetime.strptime(ref_month[0:10], '%Y-%m-%d') + relativedelta(days=(pred_month - 1)))

        result = result[result['index'] >= ref_month]
        result = result[result['index'] <= last_month]

        result = result[['index', 'yhat1', y_col]]
        result.columns = ['index', 'yhat1', 'y']
        result['sub_data'] = result['yhat1'] - result['y']


        train_data['ds'] = train_data['ds'].astype(str)
        if pred_cycle == "0":
            train_data['ds'] = train_data['ds'].str[0:7]
        elif pred_cycle == "1":
            train_data['ds'] = train_data['ds'].str[0:10]

        y = result['y']
        yhat1 = result['yhat1']

        return result, NP, y, yhat1
    #--------------------------------------------------


    #DB에서 해당 테이블 가져오기
    #결측치,
    def get_df(self, svrhost, userId, userPw, DBName, table):
        svrhost = svrhost
        userId = userId
        userPw = userPw
        DBName = DBName
        table = table

        conn = pymssql.connect(host=svrhost, user=userId, password=userPw, database=DBName)

        querys= '''
        SELECT * FROM 
        ''' + table

        dataset = pd.read_sql(sql=querys, con=conn)
        conn.close()
        return dataset

    #결측치 처리 합침
    def na_process(self, dataset, m1, rm_col, m1re, re_col):
        if m1 == 'na':
            dataset_sub = pd.DataFrame(columns = rm_col.split(","))
            dataset.dropna(subset = dataset_sub , axis = 0, inplace = True)
        if m1re == 're':
            re_col = re_col.replace("'","")
            re_col = list(re_col.split(","))
            for i in range(0, len(re_col)-1, 2):
                dataset[re_col[i]] = dataset[re_col[i]].fillna(re_col[i+1])
        return dataset

    # #컬럼별 데이터 파싱해서 사용
    # def na_process_new(self, dataset, na):
    #     for item in args["na"]:
    #         if item.value == "media":
    #             data_sub = pd.Dataframe(columns = item.key ) # median 처리
    #         elif item.value
    #
    #     return dataset


    def outlier_process(self, dataset, m2):
        if m2 =='outlier':
            quartile_1 = dataset.quantile(0.25)
            quartile_3 = dataset.quantile(0.75)
            IQR = quartile_3 - quartile_1
            condition = (dataset < (quartile_1 - 1.5 * IQR)) | (dataset > (quartile_3 + 1.5 * IQR))
            condition = condition.any(axis=1)
            dataset = dataset[condition]
        return dataset

    #xdatas -> feature 변경
    #ydata -> target 변경
    def col_name(self, feature, target):
        feature = feature.replace("'", "")
        target = target.replace("'", "")
        feature = feature.replace("[", "")
        x_col = feature.replace("]", "")
        target = target.replace("[", "")
        y_col = target.replace("]", "")
        return x_col, y_col

    def col(self,feature, target):
        x_col = feature.replace("'", "")
        y_col = target.replace("'", "")
        return x_col, y_col

    def df_select(self, dataset, x_col, y_col):
        X = pd.DataFrame(dataset, columns=x_col.split(","))
        y = pd.DataFrame(dataset, columns=y_col.split(","))
        return X, y

    def df_scaling(self, dataset, col, scaling):
        if scaling == 'none':
            return dataset
        elif scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'min-max':
            scaler = MinMaxScaler()

        col = col.split(",")
        for c in col:
            dataset[[c]] = scaler.fit_transform(dataset[[c]])

        return dataset

    #test_size, random_state 인자 받아서 적용할 수 있도록 변경
    def reg_df_split(self, X, y, x_col, y_col, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=float(test_size),
                                                            random_state=int(random_state))
        return X_train, y_train, X_test, y_test


    def class_df_split(self, X, y, x_col, y_col, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=float(test_size),
                                                            stratify = y,
                                                            random_state=int(random_state))
        return X_train, y_train, X_test, y_test

    def reg_metrics1(self, y_test, y_predict):

        # predict가 음수로 나온 경우 0으로 대체
        y_predict.loc[y_predict < 0] = 0

        rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        r2 = r2_score(y_test, y_predict)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))

        # 실제 값이 0일때 mape가 무한대로 커지는 것을 방지
        y_test.loc[y_test == 0] = 0.001
        mape = mean_absolute_percentage_error(y_test, y_predict)
        reg_metrics = {"rmse": float(rmse), "r2": float(r2), "mape": float(mape), "rmsle": float(rmsle)}

        return reg_metrics
    
    def reg_metrics(self, y_test, y_predict, y_col):
        # predict가 음수로 나온 경우 0으로 대체
        # y_predict = pd.DataFrame(y_predict, columns=[y_col])
        # y_predict.loc[y_predict[y_col] < 0, y_col] = 0

        rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        r2 = r2_score(y_test, y_predict)
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_predict))

        # 실제 값이 0일때 mape가 무한대로 커지는 것을 방지
       # y_test.loc[y_test[y_col] == 0, y_col] = 0.001
        mape = mean_absolute_percentage_error(y_test, y_predict)
        reg_metrics = {"rmse": float(rmse), "r2": float(r2), "mape": float(mape), "rmsle": float(rmsle)}

        return reg_metrics

        # 평가 지표 종류별로 함수 작성해서 필요한 모델에 따라 호출해서 원하는 평가 지표 값 받게끔 작성

    def class_metrics(self, y_test, y_predict):
        class_metrics = classification_report(y_test, y_predict, output_dict=True)

        return class_metrics

    #time_series chart그리기
    def draw_time_series_chart(self, dataset, X_col, y_col, yhat_col, model_id, algorithm_name):
        dataset = dataset.set_index(X_col)
        path = self.IMG_PATH + algorithm_name + "/"

        try:
            fig = plt.figure(figsize=(30, 10))
            fig.set_facecolor('white')
            plt.plot(dataset[y_col])
            plt.plot(dataset[yhat_col], color = 'r')

            current_values = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:,.2f}'.format(x) for x in current_values])
            plt.savefig(path + model_id, format='png', dpi=300)
            plt.close()
            msg = "Success"
        except:
            msg = "Fail"
        return msg

#수정 algorithm_name 필요해서 넣음
    def save_model(self, model, algorithm_name, algorithm_id, create_time_string, save_as_pkl):
        """
        학습한 모델을 파일 형태로 저장하고 DB에 insert할 model_id와 model_path를 return
        :param model: 저장할 모델
        :param algorithm_name: 모델 경로에 쓰일 알고리즘 이름 Ex) random_forest, cnn, ...
        :param algorithm_id: c# argument로 받은 algorithm_id
        :param create_time_string: c# argument로 받은 모델 생성일자
        :param save_as_pkl: true일 시 pkl로 저장(머신러닝 모델), false일시 h5로 저장(딥러닝 모델)
        :return: al_id+create_time으로 생성한 model_id, model을 저장한 경로인 model_path
        """
        if os.path.isdir(self.MODEL_BASE_PATH) is False:
            os.mkdir(self.MODEL_BASE_PATH)

        model_path = self.MODEL_BASE_PATH + '/' + algorithm_name
        model_id = algorithm_id + '_' + create_time_string

        if os.path.isdir(model_path) is False:
            os.mkdir(model_path)

        if save_as_pkl:
            joblib.dump(model, model_path + '/' + model_id + '.pkl')
        else:
            model.save(model_path + '/' + model_id + '.h5')

        return model_id, model_path, algorithm_name

    def apply_model(self, path, x_dataset):
        model = joblib.load(path)
        result = model.predict(x_dataset)
        return result

    # 일간 집계
    def daily_accu(self, dataset, x_col, y_col):
        dataset[y_col] = pd.to_numeric(dataset[y_col])
        dataset[x_col] = dataset[x_col].astype(str)
        dataset[x_col] = dataset[x_col].str[0:10]
        dataset = dataset.groupby(by=x_col, as_index=False).sum()
        return dataset

    # 월간 집계
    def monthly_accu(self, dataset, x_col, y_col):
        dataset[y_col] = pd.to_numeric(dataset[y_col])
        dataset[x_col] = dataset[x_col].astype(str)
        dataset[x_col] = dataset[x_col].str[0:7]
        dataset = dataset.groupby(by=x_col, as_index=False).sum()
        return dataset

    # dataset 기준일 이전데이터로 자르기
    # 기준일 이전 데이터가 없거나 비어있는 경우 0으로 메꿔줌
    def fill_daily_na(self, dataset, x_col, y_col, fin_date):
        fin_date = str(datetime.strptime(fin_date[0:10], '%Y-%m-%d') + relativedelta(days=-1))
        last_date = dataset[x_col].iloc[-1]
        if last_date < fin_date:
            new_row = [fin_date]
            for i in range(len(dataset.columns) - 1):
                new_row.append(0)
            new_row = tuple(new_row)
            add_row = [new_row]
            add_df = pd.DataFrame(add_row, columns=[x_col, y_col])
            dataset = dataset.append(add_df, ignore_index=True)

        dataset[x_col] = pd.to_datetime(dataset[x_col])
        dataset = dataset.set_index(x_col)
        dataset = dataset.asfreq('D')
        dataset = dataset.fillna(0)
        dataset = dataset.reset_index()
        dataset = dataset.loc[dataset[x_col] <= fin_date]
        return dataset

    # dataset 기준연월 이전데이터로 자르기
    # 예측일이 마지막 연월보다 한 달 이상 차이날 경우 그 사이 월은 0으로 메꿔줌
    def fill_monthly_na(self, dataset, x_col, y_col, fin_date):
        fin_date = str(datetime.strptime(fin_date[0:7], '%Y-%m') + relativedelta(months=-1))
        last_date = dataset[x_col].iloc[-1]
        while last_date < fin_date:
            next_month = str(datetime.strptime(last_date, "%Y-%m") + relativedelta(months=1))[0:7]
            new_row = [next_month]
            for i in range(len(dataset.columns) - 1):
                new_row.append(0)
            new_row = tuple(new_row)
            add_row = [new_row]
            add_df = pd.DataFrame(add_row, columns=[x_col, y_col])
            dataset = dataset.append(add_df, ignore_index=True)
            last_date = dataset[x_col].iloc[-1]

        dataset = dataset.loc[dataset[x_col] <= fin_date]
        return dataset






