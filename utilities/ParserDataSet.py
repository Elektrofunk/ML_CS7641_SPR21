from imblearn.over_sampling import RandomOverSampler
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
cf.go_offline()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
from pylab import savefig
class ParseDataSet:

    def __init__(self, raw_df, y_col_name, scale_data='no', remove_outliers=False, over_sample=False, test_size=0.3,
                 random_state=0,feature_encode = False, stratify = False, data_set_figs=False):
        if data_set_figs:
            fig = raw_df[y_col_name].value_counts().iplot(asFigure=True, kind='bar', barmode='stack',
                                               xTitle='Quality Value', yTitle='Counts', title='Class Output original vs Count')
            fig.show()
            fig.write_image('label_class_count' + '.png')
            print(raw_df[y_col_name].value_counts())
            corr = raw_df.corr()

            sns.set(font_scale=0.75)
            plt.figure(figsize=(35, 15))

            sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True, annot=True, cmap='YlGnBu', linecolor="black")
            plt.title('Correlation between features');
            plt.savefig('original_heatmap.png',dpi=600)
        self.DataWrangling = {}
        self.y = raw_df[y_col_name].copy()#.to_frame()
        self.X = raw_df.drop(columns=y_col_name)
        self.stratify = stratify
        strat_ifY = None
        if self.stratify == True:
            strat_ifY = self.y

        self.XTrain, self.XTest, self.yTrain, self.yTest = model_selection.train_test_split(
            self.X, self.y, test_size=test_size, random_state=0, stratify = strat_ifY)

        #Use this when your set only has categorical features
        if feature_encode:
            self.XTrain = pd.get_dummies(self.XTrain, prefix_sep='_')
            self.XTest = pd.get_dummies(self.XTrain, prefix_sep='_')

        # All post processing of the data set is done AFTER we split between train and test sets.
        # https://www.xspdf.com/help/50480465.html
        if remove_outliers:
            Q1 = self.XTrain.quantile(0.25)
            Q3 = self.XTrain.quantile(0.75)
            IQR = Q3 - Q1
            idx = ~((self.XTrain < (Q1 - 1.5 * IQR))
                                        | (self.XTrain > (Q3 + 1.5 * IQR))).any(axis=1)
            self.XTrain = self.XTrain[idx]
            self.yTrain = self.yTrain[idx]
            fig = self.yTrain.value_counts().iplot(asFigure=True, kind='bar', barmode='stack',
                                                      xTitle='Quality Value', yTitle='Counts',
                                                      title='Class Output after removal of outliers vs Count')
            fig.show()
            fig.write_image('label_class_count_aft_outlier_remove' + '.png')
        # https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
        if over_sample == True:
            oversample = RandomOverSampler(sampling_strategy='minority')
            self.XTrain, self.yTrain = oversample.fit_resample(self.XTrain, self.yTrain)
            if data_set_figs:
                fig = self.yTrain.value_counts().iplot(asFigure=True, kind='bar', barmode='stack',
                                                          xTitle='Quality Value', yTitle='Counts',
                                                          title='Class Output after oversample vs Count')
                fig .show()
                fig.write_image('label_class_count_aft_oversample' + '.png')
        #         if scale_data == 'min_max':
        #             pass
        # https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
        #https: // datascience.stackexchange.com / questions / 27615 / should - we - apply - normalization - to - test - data -as-well
        if scale_data == 'standard':
            scaler = StandardScaler()
            self.XTrain = scaler.fit_transform(self.XTrain)
            self.XTest = scaler.transform(self.XTest)
        if data_set_figs:
            sns.heatmap(corr, vmax=.8, linewidths=0.01, square=True, annot=True, cmap='YlGnBu', linecolor="black")
            plt.title('Correlation between features');
            plt.savefig('final_heatmap.png',dpi=1400)

            fig = self.yTest.value_counts().iplot(asFigure=True, kind='bar', barmode='stack',
                                                   xTitle='Quality Value', yTitle='Counts',
                                                   title='Class Output of test set')
            fig.write_image('yTest_Count' + '.png')
            print(self.yTest.value_counts())

    def setYTo(self, value_list, bool_input):
        self.yTrain.replace(value_list, bool_input, inplace = True)
        self.yTest.replace(value_list, bool_input, inplace = True)

    def removeFeatures(self, column_list):
        # list like ['fixed acidity','citric acid','free sulfur dioxide','chlorides']
        self.XTrain.drop(columns=column_list, inplace=True)
        self.XTest.drop(columns=column_list, inplace=True)

    def oneHotEncode(self):
        self.yTrain = pd.get_dummies(self.yTrain)
        self.yTest = pd.get_dummies(self.yTest)

    def getCVSubset(self):
        if self.stratify == True:
            strat_ifY = self.yTrain
        return model_selection.train_test_split(self.XTrain, self.yTrain, test_size=0.3, stratify= strat_ifY)

    def setDataWrangling(self, dr):
        self.DataWrangling = dr

    def getDataWrangling(self):
        return self.DataWrangling