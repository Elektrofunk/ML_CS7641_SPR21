from sklearn.model_selection import learning_curve
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt # drawing graphs
from utilities.cf_matrix import make_confusion_matrix as cf_m
from scipy import signal, misc

#source https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score, \
    plot_confusion_matrix
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
import time
import pickle
import sys
from datetime import datetime
class EstimatedEstimator:

    def __init__(self,estimator,name="john_doe", verbose = False):
        self.name = name
        self.estimator = estimator
        self.learn_curve = self.LearningCurveParams()
        self.X = np.zeros([2,2])
        self.y = np.zeros([2,2])
        self.y_pred = None
        self.yTest = None
        self.train_time = 0
        self.prediction_time = 0
        self.model_size_bytes = 0
        self.report = {}
        self.verbose = verbose

    def set_training_data(self, X, y):

        self.X = X
        self.y = y

    # def train_estimator(self):
    #     self.estimator = self.estimator.fit(self.X, self.y)

    def fit_model(self, xTrain, xTest, yTrain, yTest):
        start_fit = time.time()
        self.estimator.fit(xTrain, yTrain)
        end_fit = time.time()
        start_pred = time.time()
        self.y_pred = self.estimator.predict(xTest)
        end_pred = time.time()
        self.yTest = yTest
        #metrics
        self.train_time = end_fit - start_fit
        self.prediction_time = end_pred - start_pred
        #https: // stackoverflow.com / questions / 45601897 / how - to - calculate - the - actual - size - of - a - fit - trained - model - in -sklearn
        p = pickle.dumps(self.estimator)

        self.model_size_bytes = sys.getsizeof(p)
        now = datetime.now()
        self.fit_date =now.strftime("%Y%m%d_%H_%M_%S")

    # only do this after you have candidates
    def set_test_predict_predictions(self, xTest, yTest):
        self.y_pred = self.estimator.predict(xTest)
        self.yTest = yTest

    def set_learning_curve_params(self, cv, n_jobs, train_sizes=np.linspace(.1, 1.0, 20)):

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(copy.deepcopy(self.estimator), self.X, self.y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        self.learn_curve.update(train_scores_mean,train_scores_std,test_scores_mean,test_scores_std,fit_times_mean,fit_times_std, train_sizes)

    #https: // scikit - learn.org / stable / auto_examples / model_selection / plot_learning_curve.html
    def plot_learning_curve(self, axes=None, ylim=None, title=""):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.

        Parameters
        ----------
        estimator : estimator instance
            An estimator instance implementing `fit` and `predict` methods which
            will be cloned for each validation.

        title : str
            Title for the chart.

        X : array-like of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : array-like of shape (n_samples) or (n_samples, n_features)
            Target relative to ``X`` for classification or regression;
            None for unsupervised learning.

        axes : array-like of shape (3,), default=None
            Axes to use for plotting the curves.

        ylim : tuple of shape (2,), default=None
            Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 5-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like of shape (n_ticks,)
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the ``dtype`` is float, it is regarded
            as a fraction of the maximum size of the training set (that is
            determined by the selected validation method), i.e. it has to be within
            (0, 1]. Otherwise it is interpreted as absolute sizes of the training
            sets. Note that for classification the number of samples usually have
            to be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        axes = axes[:]
        ylim = (0.7, 1.01)
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")


        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(self.learn_curve.train_sizes, self.learn_curve.train_scores_mean - self.learn_curve.train_scores_std,
                             self.learn_curve.train_scores_mean + self.learn_curve.train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(self.learn_curve.train_sizes, self.learn_curve.test_scores_mean - self.learn_curve.test_scores_std,
                             self.learn_curve.test_scores_mean + self.learn_curve.test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(self.learn_curve.train_sizes, self.learn_curve.train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(self.learn_curve.train_sizes, self.learn_curve.test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(self.learn_curve.train_sizes, self.learn_curve.fit_times_mean, 'o-')
        axes[1].fill_between(self.learn_curve.train_sizes, self.learn_curve.fit_times_mean - self.learn_curve.fit_times_std,
                             self.learn_curve.fit_times_mean + self.learn_curve.fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(self.learn_curve.fit_times_mean, self.learn_curve.test_scores_mean, 'o-')
        axes[2].fill_between(self.learn_curve.fit_times_mean, self.learn_curve.test_scores_mean - self.learn_curve.test_scores_std,
                             self.learn_curve.test_scores_mean + self.learn_curve.test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        plt.savefig(title+'_learning_curve.png', dpi=600)
        if self.verbose:
            plt.show()


    def single_learning_curve_plot(self, title,subtitle ="",comment =""):

        fig = make_subplots(rows=1, cols=1)

        fig.append_trace(go.Scatter(
            x=self.learn_curve.train_sizes,
            y=self.learn_curve.train_scores_mean, name='training curve', mode='lines'
        ), row=1, col=1)
        #https: // stackoverflow.com / questions / 38274102 / is -there - any - way - to - make - a - plotly - scatter - have - smooth - lines - connecting - points
        fig.add_trace(go.Scatter(x=self.learn_curve.train_sizes, y=self.learn_curve.test_scores_mean,
                                 mode='lines',
                                 line = {'shape': 'spline', 'smoothing': 1.3},
                                 name='cross validation curve'))

        fig.update_layout(
            title=title+"<br>"+subtitle+"<br>"+comment,
            xaxis_title="training instances",
            yaxis_title="score",
            legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=10,
                color="RebeccaPurple"
            ))
        # fig.update_layout(title=go.layout.Title(text=subtitle, font=dict(
        #     family="Courier New, monospace",
        #     size=6,
        #     color="#0000FF"
        # )))
        #https: // stackoverflow.com / questions / 58166002 / how - to - add - caption - subtitle - using - plotly - method - in -python
        # fig.update_layout(annotations=[
        #     go.layout.Annotation(
        #         showarrow=False,
        #         text=subtitle,
        #         xanchor='center',
        #         xshift=5,
        #         yanchor='middle',
        #         font=dict(
        #             family="Courier New, monospace",
        #             size=6,
        #             color="#0000FF"
        #         )
        #     )])

        #fig['layout']['xaxis'].update(side='top')
        if self.verbose:
            fig.show()
        fig.write_image(title +'_learnc'+'.png')

    def plot_confusion_matrix(self, multi = False, title = ""):
        # estim = copy.deepcopy(self.estimator)
        # estim.fit(xTrain,yTrain)
        # y_pred = estim.predict(xTest)
        #https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        cf_matrix = confusion_matrix(self.yTest, self.y_pred)
        if multi == True:
            labels = ['TrueNeg', 'False Pos', 'FalseNeg', 'True Pos']
            categories = ['Zero', 'One']
            conf_matrix = cf_m(cf_matrix, group_names=labels, categories=categories, xyplotlabels=False, sum_stats=True ,
                        figsize=(14, 14), title = title)
        else:
            conf_matrix =  cf_m(cf_matrix, xyplotlabels=False , sum_stats=True,
                    figsize=(14, 14), title = title)
        return conf_matrix

    def get_confusion_matrix(self):
        # estim = copy.deepcopy(self.estimator)
        # estim.fit(xTrain,yTrain)
        # y_pred = estim.predict(xTest)
        #https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
        cf_matrix = confusion_matrix(self.yTest, self.y_pred)
        return cf_matrix

    def set_report(self, report):
        self.report = report

    def get_report(self):
        return self.report

#https://plotly.com/python/roc-and-pr-curves/
    def plot_roc_curve(self, title):

        # estim = copy.deepcopy(self.estimator)
        # estim.fit(xTrain, yTrain)
        # y_pred = estim.predict(xTest)

        fpr, tpr, thresholds = roc_curve(self.yTest, self.y_pred, drop_intermediate= False)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})<br>'+title,
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        if self.verbose:
            fig.show()
        fig.write_image(title+'_roc'+'.png')
        return fpr, tpr, thresholds, auc(fpr,tpr)


    class LearningCurveParams:
        def update(self,train_scores_mean, train_scores_std, test_scores_mean, test_scores_std, fit_times_mean,fit_times_std,train_sizes):
            self.train_scores_mean = train_scores_mean
            self.train_scores_std = train_scores_std
            self.test_scores_mean = test_scores_mean
            self.test_scores_std = test_scores_std
            self.fit_times_mean = fit_times_mean
            self.fit_times_std = fit_times_std
            self.train_sizes = train_sizes
