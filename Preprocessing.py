from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

# 결측치 대체, Label_encoding등 컬럼에 대한 전처리 클래스
class Preprocessing():
    # 컬럼들 안의 이상 값들
    na_values = ['$', '#VALUE!', '##', 'XNA', '@', '#', 'x', '&']
    # float데이터 인데, object로 되어있는 이상 컬럼들
    columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Score_Source_3', 'Population_Region_Relative', 'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days']

    def __init__(self):
        onehot_encoder = OneHotEncoder()
        label_encoder = LabelEncoder()

        # csv파일 불러오는 과정 na_values안에 포함된 것들은 결측치로 대체
        self.__pay_df = pd.read_csv('./nbfi_vehicle_loan_repayment_dataset/Train_Dataset.csv', na_values=self.na_values, encoding='utf-8', engine='python')
        # columns에 object로 되어있는 컬럼들을 float으로 변경
        for column in self.columns:
            self.__pay_df[column] = pd.to_numeric(self.__pay_df[column], errors='coerce')
        # target data 분리
        self.__y_target = self.__pay_df['Default']
        del self.__pay_df['Default']

    # 결측치를 어떻게 할 것인가
    def missing_value(self):
        # object인 column들(카테고리)만 뽑기
        object_columns = self.__pay_df.select_dtypes(include='object').columns
        # object가 아닌 column들(연속형)만 뽑기
        not_object_columns = self.__pay_df.select_dtypes(exclude='object').columns
        # 연속형 데이터의 평균값으로 결측치 대체
        for column in not_object_columns:
            self.__pay_df[column] = self.__pay_df[column].fillna(self.__pay_df[column].mean())

    # data를 리턴하는 함수
    def get_df(self):
        return self.__pay_df
    
    # target_df를 리턴하는 함수
    def get_target_df(self):
        return self.__y_target