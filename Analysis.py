#
# Builds XGBoost & Polynomial based models and generates results.html file
#
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
import xgboost as xgb
from DocGen import *

DATA = pd.DataFrame()
##
# Prepares the data and fills missing values using polynomial interpolation
##
def prepareData(filename, fitOrder, newColumnNames):
    global DATA
    csv = pd.read_csv(filename)
    # print(csv.head())

    csv.iloc[:, 1] = csv.iloc[:, 1].interpolate(method='polynomial', order=fitOrder)

    for i in range(1, len(csv)):
        if math.isnan(csv.iloc[i, 2]):
            csv.iloc[i, 2] = (1 -  csv.iloc[i-1, 1] / csv.iloc[i, 1]) * 100

    csv.columns = newColumnNames
    #print(csv.loc[(csv['date'] >= '1940-12-31') & (csv['date'] < '1955-12-31')])

    if (DATA.empty):
        DATA = csv
    else:
        DATA = pd.merge(DATA, csv, on='date')

    # inspectDataframe(DATA)
    return csv
#
# Function to print summary of dataframe
#
def inspectDataframe(df):
    print('-' * 80)
    print('Columns: {}'.format(df.columns))
    print('Shape: {}'.format(df.shape))
    print(df.head())
    print(df.tail())

#
# Gets the subset of data from dataframe between startYear & endYear
#
def getDataUptoYear(data, startYear = 0, endYear = 2020):
    data['Year'] = pd.DatetimeIndex(data['date']).year
    return data.loc[(data['Year'] > startYear) & (data['Year'] <= endYear)]

#
# Function to plot comparison of population againts various datapoints
#
def plotComparison(data):
    addTitle('Comparison')
    addLine()
    
    path = 'images/Comparison/'
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.scatter(data['Deaths'], data['Population'])
    plt.grid(True)
    plt.title('Deaths vs Population')
    plt.xlabel('Deaths Per 1000')
    plt.ylabel('Population')
    file_name = path + 'Deaths_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Deaths vs Population')


    plt.scatter(data['Births'], data['Population'])
    plt.grid(True)
    plt.title('Births vs Population')    
    plt.ylabel('Population')
    file_name = path + 'Births_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Births vs Population')

    plt.scatter(data['InfantMortality'], data['Population'])
    plt.grid(True)
    plt.title('Infant Mortality vs Population')
    plt.ylabel('Population')
    file_name = path + 'InfantMortality_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Infant Mortality vs Population')

    plt.scatter(data['LifeExpectancy'], data['Population'])
    plt.grid(True)
    plt.title('Life Expectancy vs Population')
    plt.ylabel('Population')
    file_name = path + 'LifeExpectancy_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Life Expectancy vs Population')

    plt.scatter(data['BirthsPerWoman'], data['Population'])
    plt.grid(True)
    plt.title('Fertility vs Population')
    plt.ylabel('Population')
    file_name = path + 'FertilityRate_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Fertility vs Population')

    plt.scatter(data['MigrantPopulation'], data['Population'])
    plt.grid(True)
    plt.title('Immigration vs Population')
    plt.ylabel('Population')
    file_name = path + 'Immigration_vs_Population.png'
    plt.savefig(file_name)
    plt.close()    
    addImageToDoc(file_name, 'Immigration vs Population')

    plt.scatter(data['NetMigration'], data['Population'])
    plt.grid(True)
    plt.title('Net Migration vs Population')
    plt.ylabel('Population')
    file_name = path + 'NetMigration_vs_Population.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Net Migration vs Population')
    addLine()

#
# Helper function to add image to html doc for reporting
#
def addImageToDoc(file_name, title):
    addSubTitle(title)
    addImage(file_name, title)

#
# Combines training & test data with results for reporting
#
def combineResults(
        X_train, 
        y_train, 
        pred_train, 
        X_test, 
        y_test, 
        predictions,         
        predictedColumnName):
    pdTrain = pd.DataFrame(pred_train, columns=[predictedColumnName]) 

    trainingValues = pd.merge(X_train, y_train, left_index=True, right_index=True)
    trainingValues = pd.merge(trainingValues.reset_index(drop=True), pdTrain, left_index=True, right_index=True)

    pdTest = pd.DataFrame(predictions, columns=[predictedColumnName]) 
    testValues = pd.merge(X_test, y_test, left_index=True, right_index=True)
    testValues = pd.merge(testValues.reset_index(drop=True), pdTest, left_index=True, right_index=True)

    values = pd.concat([trainingValues, testValues])
    values = values.sort_values("Year")
    return values

#
# Generates summary from predictions
#
def generateSummary(
    columnToPredict, 
    path, 
    X_train, X_test,
    y_train, y_test,
    predictions, pred_train,
    score, predictedColumnName):
    values = combineResults(
        X_train, 
        y_train, 
        pred_train, 
        X_test, 
        y_test, 
        predictions, 
        predictedColumnName)

    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(values.Year, values[columnToPredict], 'y', linewidth=5, label='Actual')
    plt.plot(values.Year, values[predictedColumnName], 'b', linewidth=1, label='Predicted')
    plt.legend(loc="upper left")
    plt.title("Actual vs Predicted")
    file_name = path + 'Actual_vs_Predicted.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Actual vs Predicted')

    addSubTitle('Data Dimensions')
    startTable()
    addHeaderRow(['Training Data Size', 'Test Data Size'])
    addRow([str(X_train.shape[0]), str(X_test.shape[0])])
    endTable()

    addSubTitle('Test Results')
    startTable()
    addHeaderRow(['#', 'Data Point', 'Actual', 'Prediction'])
    for i in range(0, X_test.shape[0]):
        addRow([
            str(i + 1),
            X_test.iloc[i, :].to_string(header=True, index=True, float_format=lambda x: "{:.1f}".format(x)), 
            '{:.2f}'.format(round(y_test.iloc[i], 2)), 
            '{:.2f}'.format(round(predictions[i], 2))])
    endTable()

    addSubTitle('Error Calculations') 

    startTable()
    addHeaderRow(['R-squared', 'Mean-squared Error', 'Mean-squared Error for Predictions', 'Model Score'])
    addRow([
        '{:.2f}'.format(r2_score(y_train, pred_train)),
        '{:.5f}'.format(mean_squared_error(y_train, pred_train)),
        '{:.5f}'.format(mean_squared_error(y_test, predictions)),
        '{:.2f}'.format(score)
    ])
    endTable()   
    addLine()

#
# XGBoost analysis
#    
def xgbTest(df, title, columnsForAnalysis, columnToPredict):
    print ('\n\n')
    print ('*' * 80)
    print('\n{}\n\n'.format(title))
    addTitle(title)
    path = 'images/{}/'.format(title)
    if not os.path.isdir(path):
        os.makedirs(path)

    print(df.shape)
    print(df.head())

    data = df[columnsForAnalysis]

    print (data.columns)

    print(data.info())
    print(data.head())

    print (data.isna().sum())

    X = data.drop(columnToPredict, axis=1)
    y = data[columnToPredict]

    print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    print(X_train.shape)
    print(X_test.shape)

    model = XGBRegressor() #objective='reg:squarederror', eval_metric='rmse', n_estimators = 10, seed = 123)
    # model = XGBRegressor(learning_rate = 0.1, n_estimators=1000,
    #                        max_depth=5, min_child_weight=1,
    #                        gamma=0, subsample=0.8,
    #                        colsample_bytree=0.8, objective= "reg:squarederror",  
    #                        nthread=-1, scale_pos_weight=1, seed=27)

    model.fit(X_train, y_train)
    print(model)

    plt.figure(figsize=(9, 6))
    plt.tight_layout()
    plt.legend(loc="lower right")
    xgb.plot_importance(model, ax=plt.gca())
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0, hspace=0)
    file_name = path + 'Importance.png'
    plt.savefig(file_name)
    plt.close()
    addImageToDoc(file_name, 'Importance')
   
    predictions = model.predict(X_test)
    pred_train = model.predict(X_train)
    print(X_train.columns)
    print(X_train.head())
    
    score = model.score(X_test, y_test)
    predictedColumnName = '{}_Predicted'.format(columnToPredict)

    generateSummary(
        columnToPredict, 
        path, 
        X_train, 
        X_test,
        y_train, 
        y_test,
        predictions, 
        pred_train, 
        score, 
        predictedColumnName)

    return model

#
# Polynomial Analysis
#
def polynomialTest(df, title, columnsForAnalysis, columnToPredict):
    print('\n\{}\n\n'.format(title))
    path = 'images/{}/'.format(title)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    addTitle(title)
    # data = data.drop(dropColumns, axis=1)
    data = df[columnsForAnalysis]
    # print (data.columns)

    X = data.drop(columnToPredict, axis=1)
    y = data[columnToPredict]   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    poly = PolynomialFeatures(degree=3, include_bias=False)

    X_train_features = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_features, y_train)
    print(model)

    X_test_features = poly.fit_transform(X_test)
    predictions = model.predict(X_test_features)
    print(predictions)
    print('Data point 1')
    print(X_test.iloc[0])
    print('Expected: {}'.format(y_test.iloc[0]))

    print('Data point 2')
    print(X_test.iloc[1])
    print(y_test.iloc[1])

    pred_train = model.predict(X_train_features)
    score = model.score(X_test_features, y_test)
    predictedColumnName = '{}_Predicted'.format(columnToPredict)

    generateSummary(
        columnToPredict, 
        path, 
        X_train, 
        X_test,
        y_train, 
        y_test,
        predictions, 
        pred_train, 
        score, 
        predictedColumnName)

    return model

#
# SVM analysis
#
def svmTest(df, title, columnsForAnalysis, columnToPredict):
    print('\n\{}\n\n'.format(title))
    path = 'images/{}/'.format(title)
    if not os.path.isdir(path):
        os.makedirs(path)
    
    addTitle(title)
    data = df[columnsForAnalysis]

    X = data.drop(columnToPredict, axis=1)
    y = data[columnToPredict]   
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = SVR(kernel='rbf', C=100, epsilon=1)
    model.fit(X_train, y_train)
    print(model)

    predictions = model.predict(X_test)
    print(predictions)
    print('Data point 1')
    print(X_test.iloc[0])
    print('Expected: {}'.format(y_test.iloc[0]))

    print('Data point 2')
    print(X_test.iloc[1])
    print(y_test.iloc[1])

    pred_train = model.predict(X_train)

    pred_future = None

    score = model.score(X_test, y_test)
    predictedColumnName = '{}_Predicted'.format(columnToPredict)

    generateSummary(
        columnToPredict, 
        path, 
        X_train, 
        X_test,
        y_train, 
        y_test,
        predictions, 
        pred_train, 
        score, 
        predictedColumnName)

    return model

#
# Main function
#
def main():
    global DATA
    print("Population Prediction")
    
    filename = './Data/USA/{}.csv'
    polyFitOrder = 2

    prepareData(filename.format('USA-Birthrate'), polyFitOrder, ['date', 'Births', 'BirthChangePercent'])
    prepareData(filename.format('USA-DeathRate'), polyFitOrder, ['date', 'Deaths', 'DeathChangePercent'])
    prepareData(filename.format('USA-InfantMortalityRate'), polyFitOrder, ['date', 'InfantMortality', 'InfantMortalityChangePercent'])
    prepareData(filename.format('USA-LifeExpectancy'), polyFitOrder, ['date', 'LifeExpectancy', 'LifeExpectancyChangePercent'])
    prepareData(filename.format('USA-NetMigration'), polyFitOrder, ['date', 'NetMigration', 'NetMigrationChangePercent'])
    prepareData(filename.format('USA-FertilityRate'), polyFitOrder, ['date', 'BirthsPerWoman', 'BirthsPerWomanChangePercent'])
    prepareData(filename.format('USA-Immigration'), polyFitOrder, ['date', 'MigrantPopulation', 'MigrantPopulationChangePercent'])
    prepareData(filename.format('USA-Population'), polyFitOrder, ['date', 'Population', 'PopulationChangePercent'])

    inspectDataframe(DATA)

    startHtml()
    startBody()

    data = getDataUptoYear(DATA, 0, 2020)
    plotComparison(data)

    polyBirthRateModel = polynomialTest(data, 'Polynomial - Births', ['Year', 'Births'], 'Births')
    #svmBirthRateModel = svmTest(data, 'SVM - Births', ['Year', 'Births'], 'Births')
    xgbBirthRateModel = xgbTest(data, 'XGB - Births', ['Year', 'Births'], 'Births')
  
    polyDeathRateModel = polynomialTest(data, 'Polynomial - Deaths', ['Year', 'Deaths'], 'Deaths')
    #svmDeathRateModel = svmTest(data, 'SVM - Deaths', ['Year', 'Deaths'], 'Deaths')
    xgbDeathRateModel = xgbTest(data, 'XGB - Deaths', ['Year', 'Deaths'], 'Deaths')

    polyInfantMortalityRateModel = polynomialTest(data, 'Polynomial - Infant Mortality', ['Year', 'InfantMortality'], 'InfantMortality')
    #svmInfantMortalityRateModel = svmTest(data, 'SVM - Infant Mortality', ['Year', 'InfantMortality'], 'InfantMortality')
    xgbInfantMortalityRateModel = xgbTest(data, 'XGB - Infant Mortality', ['Year', 'InfantMortality'], 'InfantMortality')

    polyLifeExpectancyRateModel = polynomialTest(data, 'Polynomial - Life Expectancy', ['Year', 'LifeExpectancy'], 'LifeExpectancy')
    #svmLifeExpectancyRateModel = svmTest(data, 'SVM - Life Expectancy', ['Year', 'LifeExpectancy'], 'LifeExpectancy')
    xgbLifeExpectancyRateModel = xgbTest(data, 'XGB - Life Expectancy', ['Year', 'LifeExpectancy'], 'LifeExpectancy')

    polyNetMigrationModel = polynomialTest(data, 'Polynomial - Net Migration', ['Year', 'NetMigration'], 'NetMigration')
    #svmNetMigrationModel = svmTest(data, 'SVM - Net Migration', ['Year', 'NetMigration'], 'NetMigration')
    xgbNetMigrationModel = xgbTest(data, 'XGB - Net Migration', ['Year', 'NetMigration'], 'NetMigration')

    polyFertilityRatenModel = polynomialTest(data, 'Polynomial - Fertility', ['Year', 'BirthsPerWoman'], 'BirthsPerWoman')
    #svmFertilityRatenModel = svmTest(data, 'SVM - Fertility', ['Year', 'BirthsPerWoman'], 'BirthsPerWoman')
    xgbFertilityRatenModel = xgbTest(data, 'XGB - Fertility', ['Year', 'BirthsPerWoman'], 'BirthsPerWoman')

    polyImmigrationModel = polynomialTest(data, 'Polynomial - Immigration', ['Year', 'MigrantPopulation'], 'MigrantPopulation')
    #svmImmigrationModel = svmTest(data, 'SVM - Immigration', ['Year', 'MigrantPopulation'], 'MigrantPopulation')
    xgbImmigrationModel = xgbTest(data, 'XGB - Immigration', ['Year', 'MigrantPopulation'], 'MigrantPopulation')

    polynomialTest(data, 
        'Polynomial - Population',
        ['Year', 'Deaths', 'Births', 'InfantMortality', 'LifeExpectancy', 'BirthsPerWoman','MigrantPopulation', 'Population'],
        'Population')

    # svmTest(data, 
    #     'SVM - Population',
    #     ['Year', 'Deaths', 'Births', 'InfantMortality', 'LifeExpectancy', 'BirthsPerWoman','MigrantPopulation', 'Population'],
    #     'Population')
        
    xgbTest(data, 
        'XGB - Population', 
        ['Year', 'Deaths', 'Births', 'InfantMortality', 'LifeExpectancy', 'BirthsPerWoman','MigrantPopulation', 'Population'],
        'Population')
    
    closeBody()
    closeHtml()

    saveHtml('results.html')

if __name__ == '__main__':
    main()