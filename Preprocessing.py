from sklearn.preprocessing import  OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

# 결측치 대체, Label_encoding등 컬럼에 대한 전처리 클래스
class Preprocessing:
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

    # numerical_columns에 대해서 결측치를 어떻게 대체할 것인지        
    def numerical_columns_replace(self):
        # numerical_column들
        numerical_columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 
                   'Population_Region_Relative', 'Age_Days', 'Employed_Days',
                   'Registration_Days', 'ID_Days', 'Score_Source_1' ,
                   'Score_Source_2', 'Score_Source_3', 'Social_Circle_Default', 'Phone_Change']

        # 결측치 이상치로 대체하기
        for column in numerical_columns:
            df[column] = df[column].fillna(-999)
        


    # category_columns에 대해서 결측치를 어떻게 대체할 것인지 
    def category_columns_object_replace(self):
        # object인 column들(카테고리)만 뽑기
        category_columns_object = self.__pay_df.select_dtypes(include='object').columns
        category_columns_number = ['Car_Owned', 'Bike_Owned', 'Active_Loan', 'House_Own', 'Child_Count',
                        'Own_House_Age', 'Mobile_Tag', 'Homephone_Tag', 'Workphone_Working', 
                        'Client_Family_Members', 'Cleint_City_Rating', 'Application_Process_Day', 'Application_Process_Hour', 'Credit_Bureau',]



    # data를 리턴하는 함수
    def get_df(self):
        return self.__pay_df
    
    # target_df를 리턴하는 함수
    def get_target_df(self):
        return self.__y_target