import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import numpy as np
class DPDataPrePRocess:
    def process_data(self,datasetname):
        #dataset = pd.read_csv(datasetname)
        dataset = datasetname
        dataset = dataset[['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.dtypes)
        dataset.fillna
        dataset.dropna()
        dataset = dataset.fillna(0)
        dataset.fillna(method='ffill')
        print(dataset.isnull().values.any())
        #print(dataset.head(10))
        #plot_corr(dataset)
        #plt.show()
        print(dataset.dtypes)
        dataset.to_csv('file1.csv')
        data_dict = dataset.to_dict()
        return data_dict

    def MyLogiSticregression(self,dataset):
        print("###Logistic Regression####")
        #print('Have a great day ',dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[['DAY','DEPARTURE_TIME','FLIGHT_NUMBER','DESTINATION_AIRPORT','ORIGIN_AIRPORT','DAY_OF_WEEK','TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:,:1].values
        y = dataset.iloc[:,2].values
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3,random_state=0)

        model = LogisticRegression()
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        #acuracy = accuracy_score(y_pred,y_test)
        #print(acuracy)
        lgDict = {}
        lg_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        lg_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        lg_EVS  = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        lg_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        lg_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        lgDict.update({'lg_MAE':lg_MAE,'lg_MSE':lg_MSE,'lg_EVS':lg_EVS,'lg_MedianAE':lg_MedianAE,'lg_R2Score':lg_R2Score})

        print("MAE=",lg_MAE )
        print("MSE=",lg_MSE )
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",lg_EVS)
        print("Median Absalute Error=",lg_MedianAE)
        print("R2_Score", lg_R2Score)

        return lgDict

    def MyDecisionTree(self, dataset):
        print("###Decesion Treee####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        dtDict = {}
        dt_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        dt_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        dt_EVS =  metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        dt_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        dt_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        dtDict.update({'dt_MAE':dt_MAE,'dt_MSE':dt_MSE,'dt_EVS':dt_EVS,'dt_MedianAE':dt_MedianAE,'dt_R2Score':dt_R2Score})

        print("MAE=", dt_MAE)
        print("MSE=", dt_MAE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ",dt_EVS)
        print("Median Absalute Error=", dt_MedianAE)
        print("R2_Score", dt_R2Score)
        return dtDict

    def MyRandomForest(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        rfDict = {}
        rf_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        rf_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        rf_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        rf_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        rf_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        rfDict.update({'rf_MAE': rf_MAE, 'rf_MSE': rf_MSE, 'rf_EVS': rf_EVS, 'rf_MedianAE': rf_MedianAE,
                       'rf_R2Score': rf_R2Score})

        print("MAE=", rf_MAE)
        print("MSE=", rf_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", rf_EVS)
        print("Median Absalute Error=", rf_MedianAE)
        print("R2_Score", rf_R2Score)
        return rfDict

    def MyBayesianRidge(self, dataset):
        print("###RadomForest####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        brDict = {}
        br_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        br_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        br_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        br_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        br_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        brDict.update({'br_MAE': br_MAE, 'br_MSE': br_MSE, 'br_EVS': br_EVS, 'br_MedianAE': br_MedianAE,
                       'br_R2Score': br_R2Score})

        print("MAE=", br_MAE)
        print("MSE=", br_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", br_EVS)
        print("Median Absalute Error=", br_MedianAE)
        print("R2_Score", br_R2Score)
        return brDict

    def MyGradientBoostingRegressor(self, dataset):
        print("###GradientBoostingRegressor####")
        #print('Have a great day ', dataset)
        #dataset = pd.read_csv(dataset)
        dataset = dataset[
            ['DAY', 'DEPARTURE_TIME', 'FLIGHT_NUMBER', 'DESTINATION_AIRPORT', 'ORIGIN_AIRPORT', 'DAY_OF_WEEK',
             'TAXI_OUT']]
        #print(dataset.head())
        X = dataset.iloc[:, :1].values
        y = dataset.iloc[:, 2].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # acuracy = accuracy_score(y_pred,y_test)
        # print(acuracy)
        gbrDict = {}
        gbr_MAE = metrics.mean_absolute_error(y_pred.round(), y_test)
        gbr_MSE = metrics.mean_squared_error(y_pred.round(), y_test)
        gbr_EVS = metrics.explained_variance_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')
        gbr_MedianAE = metrics.median_absolute_error(y_test, y_pred)
        gbr_R2Score = metrics.r2_score(y_test, y_pred, sample_weight=None, multioutput='uniform_average')

        gbrDict.update({'gbr_MAE': gbr_MAE, 'gbr_MSE': gbr_MSE, 'gbr_EVS': gbr_EVS, 'gbr_MedianAE': gbr_MedianAE,
                       'gbr_R2Score': gbr_R2Score})

        print("MAE=", gbr_MAE)
        print("MSE=", gbr_MSE)
        print("RMSE=", np.sqrt(metrics.mean_squared_error(y_pred.round(), y_test)))
        print("Variance Score ", gbr_EVS)
        print("Median Absalute Error=", gbr_MedianAE)
        print("R2_Score", gbr_R2Score)
        return gbrDict




def plot_corr(data_frame, size=11):
    corr = data_frame.corr()  # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
