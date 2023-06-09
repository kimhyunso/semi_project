import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split


# 결측치 대체, Label_encoding등 컬럼에 대한 전처리 클래스
class Preprocessing:
    # 컬럼들 안의 이상 값들
    na_values = ['$', '#VALUE!', '##', 'XNA', '@', '#', 'x', '&']
    # float데이터 인데, object로 되어있는 이상 컬럼들
    columns = ['Client_Income', 'Credit_Amount', 'Loan_Annuity', 
    'Score_Source_3', 'Population_Region_Relative', 'Age_Days', 
    'Employed_Days', 'Registration_Days', 'ID_Days']

    category_columns_object = [
            'Client_Education', 'Client_Marital_Status', 
            'Client_Gender', 'Loan_Contract_Type', 'Client_Housing_Type']

    category_columns = [
            'Car_Owned', 'Bike_Owned', 'Active_Loan', 
            'House_Own', 'Homephone_Tag', 'Workphone_Working', 
            'Cleint_City_Rating', 'Application_Process_Day', 'Client_Permanent_Match_Tag', 
            'Client_Contact_Work_Tag']

    numerical_columns=[
            'Child_Count', 'Credit_Amount', 'Loan_Annuity', 
            'Population_Region_Relative', 'Age_Days', 'Employed_Days', 
            'Registration_Days', 'ID_Days', 'Client_Family_Members',
            'Score_Source_2', 'Phone_Change', 'Credit_Bureau', 'Client_Income']

    def __init__(self):
        # csv파일 불러오는 과정 na_values안에 포함된 것들은 결측치로 대체
        self.__pay_df = pd.read_csv('./nbfi_vehicle_loan_repayment_dataset/Train_Dataset.csv', na_values=self.na_values, encoding='utf-8', engine='python')

        # columns에 object로 되어있는 컬럼들을 float으로 변경
        for column in self.columns:
            self.__pay_df[column] = pd.to_numeric(self.__pay_df[column], errors='coerce')
        
        
        # target data 분리
        self.__y_target = self.__pay_df['Default']
        del self.__pay_df['Default']

        # 특정 컬럼에 대한 결측치 처리
        self.__pay_df['Client_Occupation'].fillna('Nojob', inplace=True)

    def drop_columns(self):
        drop_columns = [
            'ID', 'Own_House_Age', 'Type_Organization', 
            'Mobile_Tag', 'Score_Source_1', 'Score_Source_3', 
            'Social_Circle_Default', 'Application_Process_Hour', 'Accompany_Client', 'Client_Income_Type']

        # 삭제할 column들
        self.__pay_df.drop(columns=drop_columns, axis=1, inplace=True)

    def category_columns_replace(self):
        # 결측치 0개 초과 10000개 미만의 데이터를 대상으로 랜덤하게 결측치 대체
        for column in self.category_columns_object:
            if self.__pay_df[column].isna().sum() > 10000 or self.__pay_df[column].isna().sum() == 0:
                continue
            unique_columns = self.__pay_df[column].loc[self.__pay_df[column].isna()==False].unique()
            self.__pay_df[column] = self.__pay_df[column].apply(lambda x : random.choice(unique_columns) if pd.isna(x) else x)

        # # 나머지 범주형 데이터에 대해 one-hot encoding 적용
        # df_null_sum = self.__pay_df.isna().sum()
        # column = df_null_sum[df_null_sum>0].index
        # self.__pay_df = pd.get_dummies(self.__pay_df, columns=self.category_columns)

    # numerical_columns에 대해서 결측치를 어떻게 대체할 것인지        
    def numerical_columns_replace(self):
        # numerical_columns 평균값 대체
        for column in self.numerical_columns:
            self.__pay_df[column] = self.__pay_df[column].fillna(self.__pay_df[column].mean())

    # data를 리턴하는 함수
    def get_df(self):
        return self.__pay_df
    
    # target_df를 리턴하는 함수
    def get_target_df(self):
        return self.__y_target