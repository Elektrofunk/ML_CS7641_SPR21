from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from utilities.DatasetVis import DatasetVis as dv
from utilities.ParserDataSet import ParseDataSet as pds
from utilities.EstimatedEstimator import EstimatedEstimator as est
from sklearn.tree import DecisionTreeClassifier, plot_tree
import sys
import pickle
from utilities.Report import Report as repo
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import time

reportObj = None
import random

random.seed(10)

def test_model(model, xTest, yTest):
    test_report = {}
    model.set_test_predict_predictions(xTest, yTest)
    model.plot_confusion_matrix(multi=False, title='test_'+model.name)
    model.plot_roc_curve(title='test_'+model.name)
    test_report.update(
        dict(zip(['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score'],  model.plot_confusion_matrix( multi=False, title='test_'+model.name))))
    test_report.update(dict(zip(['test_false_positive_ratio', 'test_true_positive_ratio', 'test_thresholds', 'test_auc'],
                           model.plot_roc_curve(title='test_'+model.name))))
    return test_report

# Perform grid search for an algorithm
def general_grid_search(estimator, params, X, y, multi=False):
    import time
    # We use cost_complexity pruning to find a list of alphas for prunning over the grid(weaker to strongest links)
    start = time.time()
    # Learned gridsearch from chapter 2 of Hands-On Machine Learning with Scikit-learn.
    # source Chapter 2 Hands-On Machine Learning with Scikit-learn, Keras and TensorFlow.

    if multi == True:
        scoring = 'f1_macro'
    else:
        scoring = 'neg_mean_squared_error'

    grid_search = GridSearchCV(estimator, params, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_jobs=-1, verbose=False)

    grid_search.fit(X, y.values.ravel(order='C'))
    endtime = time.time()
    delta_time = endtime - start
    best_params = grid_search.best_params_
    # params['ccp_alpha']
    # print(grid_search.best_estimator_.feature_importances_)
    # cvres = grid_search.cv_results_
    return best_params, delta_time, grid_search


# Evaluate an algorithm
def evaluate_algo(estimator, params, parser, name="", dataset_comment="", labels=[], multi=False, use_test_set=False):
    report = {}
    report.update({'name': name})

    model = est(estimator, name=name)

    model.set_training_data(parser.XTrain, parser.yTrain)

    # CAUTION: make this flag available after you finish tweaking your model.
    if use_test_set:

        model.fit_model(parser.XTrain, parser.yTrain, parser.XTest, parser.yTest)
    else:
        xTrain, xCV, yTrain, yCV = parser.getCVSubset()
        model.fit_model(xTrain, xCV, yTrain, yCV)
    report.update({'model_fit_date': model.fit_date})
    report.update({'model_time_to_train_sec': model.train_time})
    report.update({'model_time_to_prediction_sec': model.prediction_time})
    report.update({'model_size_bytes': model.model_size_bytes})
    model.set_learning_curve_params(cv=5, n_jobs=4)  # ,train_sizes = [1,10,100,500,1119])
    model.single_learning_curve_plot(name, str(params), dataset_comment)
    model.plot_learning_curve(axes=None, ylim=None, title=name)
    xTrain, xCV, yTrain, yCV = parser.getCVSubset()
    # precission_vanilla, recall_vanilla, f1_score_vanilla = vanilla_tree.plot_confusion_matrix(xTrain, yTrain, xCV, yCV)
    if multi == False:
        report.update(
            dict(zip(['accuracy', 'precision', 'recall', 'f1_score'], model.plot_confusion_matrix(title=name))))
        report.update(dict(zip(['false_positive_ratio', 'true_positive_ratio', 'thresholds', 'auc'],
                               model.plot_roc_curve(name))))
    else:
        report.update(
            dict(zip(['accuracy', 'precision', 'recall', 'f1_score'],
                     model.plot_confusion_matrix(multi=True, title=name))))
    report.update({'params': str(params)})
    return report, model


def do_data_wrangling_for_wine():
    data_wrangling = {'data_set_name': 'wine'}
    data_wrangling['scale_data'] = True
    data_wrangling['remove_outliers'] = False
    data_wrangling['over_sample'] = False
    data_wrangling['random_state'] = 0
    if data_wrangling['data_set_name'] is 'thyroid':
        data_wrangling['label_name'] = 'target'
    else:
        data_wrangling['label_name'] = 'quality'
    data_wrangling['test_size'] = 0.3
    data_wrangling['multi'] = False
    data_wrangling['feature_encode'] = False
    data_wrangling['sep'] = ';'
    data_wrangling['stratify'] = True
    data_wrangling['data_set_figs'] = False
    data_wrangling['input_dataset'] = os.getcwd() + '/datasets/' \
                                      + data_wrangling['data_set_name'] \
                                      + '/' + data_wrangling['data_set_name'] + '.csv'

    rw_df = pd.read_csv(data_wrangling['input_dataset'], sep=data_wrangling['sep'])
    # rw_df = pd.get_dummies(rw_df)
    parser_mushroom = pds(rw_df, data_wrangling['label_name'], data_wrangling['scale_data'],
                          remove_outliers=data_wrangling['remove_outliers'],
                          over_sample=data_wrangling['over_sample'], test_size=data_wrangling['test_size'],
                          random_state=data_wrangling['random_state'], feature_encode=data_wrangling['feature_encode'],
                          stratify=data_wrangling['stratify'],
                          data_set_figs=data_wrangling['data_set_figs'])  # onehot encoding test.
    # parser_red_wine.oneHotEncode()
    parser_mushroom.setYTo([3, 4, 5], 0)
    parser_mushroom.setYTo([6, 7, 8], 1)
    remove_features = True
    if remove_features:
        parser_mushroom.removeFeatures(['fixed acidity', 'citric acid', 'free sulfur dioxide', 'chlorides'])

    parser_mushroom.setDataWrangling(data_wrangling)
    return data_wrangling, parser_mushroom


def do_data_wrangling_for_thyroid():
    data_wrangling = {'data_set_name': 'thyroid'}
    data_wrangling['scale_data'] = False
    data_wrangling['remove_outliers'] = False
    data_wrangling['over_sample'] = False
    data_wrangling['random_state'] = 0
    if data_wrangling['data_set_name'] is 'thyroid':
        data_wrangling['label_name'] = 'target'
    else:
        data_wrangling['label_name'] = 'quality'
    data_wrangling['test_size'] = 0.3
    data_wrangling['multi'] = False
    data_wrangling['feature_encode'] = False
    data_wrangling['sep'] = ','
    data_wrangling['stratify'] = True
    data_wrangling['data_set_figs'] = False
    data_wrangling['input_dataset'] = os.getcwd() + '/datasets/' + data_wrangling['data_set_name'] + '/' + \
                                      data_wrangling[
                                          'data_set_name'] + '.csv'

    rw_df = pd.read_csv(data_wrangling['input_dataset'], sep=data_wrangling['sep'])
    # rw_df = pd.get_dummies(rw_df)
    parser_mushroom = pds(rw_df, data_wrangling['label_name'], data_wrangling['scale_data'],
                          remove_outliers=data_wrangling['remove_outliers'],
                          over_sample=data_wrangling['over_sample'], test_size=data_wrangling['test_size'],
                          random_state=data_wrangling['random_state'], feature_encode=data_wrangling['feature_encode'],
                          stratify=data_wrangling['stratify'],
                          data_set_figs=data_wrangling['data_set_figs'])  # onehot encoding test.
    # parser_red_wine.oneHotEncode()
    parser_mushroom.setYTo(['negative'], 0)
    parser_mushroom.setYTo(['hypothyroid'], 1)
    parser_mushroom.setDataWrangling(data_wrangling)
    return data_wrangling, parser_mushroom


def pickle_and_move(name, object):
    we_can_pickle_that(name, object)
    reportObj.move_report_files()


def we_can_pickle_that(name, object):
    outfile = open(name + '.pickle', 'wb')
    pickle.dump(object, outfile)
    outfile.close()
    time.sleep(5)
    # reportObj.move_report_files()


if __name__ == "__main__":
    test_report_list =[]
    type = 'wine'
    run_dict = {'parser':
                    {'type': type, 'run': True, 'pickle': type+'_parser.pickle'},
                'vanilla_tree':
                    {'run': False, 'pickle': type+'_vanilla_tree.pickle', 'name': 'vanilla_tree', 'test': False},
                'grid_tree':
                    {'run': True, 'pickle':  type+'_grid_tree.pickle', 'name': 'grid_tree', 'test': False},
                'grid_tree2':
                    {'run': True, 'pickle': type + '_grid_tree2.pickle', 'name': 'grid_tree2', 'test': False},
                'vanilla_knn':
                    {'run': False, 'pickle': type+'_vanilla_knn.pickle', 'name': 'vanilla_knn', 'test': False},
                'grid_knn':
                    {'run': True, 'pickle':  type+'_grid_knn.pickle', 'name': 'grid_knn', 'test': False},
                'vanilla_ann':
                    {'run': False, 'pickle':  type+'_vanilla_ann.pickle', 'name': 'vanilla_ann', 'test': False},
                'grid_ann':
                    {'run': True, 'pickle':  type+'_grid_ann.pickle', 'name': 'grid_ann', 'test': False},
                'grid_boost':
                    {'run': False, 'pickle':  type+'_grid_boost.pickle', 'name': 'grid_boost', 'test': False},
                'grid_svm':
                    {'run': True, 'pickle':  type+'_grid_svm.pickle', 'name': 'grid_svm', 'test': False},
                'grid_svm2':
                    {'run': True, 'pickle':  type+'_grid_svm2.pickle', 'name': 'grid_svm2', 'test': False}
                }
    if run_dict['parser']['run']:

        if run_dict['parser']['type'] is 'wine':
            data_wrangling, parser_mushroom = do_data_wrangling_for_wine()
        elif run_dict['parser']['type'] is 'thyroid':
            data_wrangling, parser_mushroom = do_data_wrangling_for_thyroid()
    elif run_dict['parser']['pickle'] is not None:

        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['parser']['pickle'])
        print(path)
        parser_mushroom = pickle.load(open(path, 'rb'))
        data_wrangling = parser_mushroom.getDataWrangling()
        # report_list.append(parser_mushroom.get_report())
    # data_wrangling, parser_mushroom =do_data_wrangling_for_thyroid()

    we_can_pickle_that(data_wrangling['data_set_name'] + "_parser", parser_mushroom)
    reportObj = repo(data_wrangling['data_set_name'])
    pickle_and_move(data_wrangling['data_set_name'] + "_parser", parser_mushroom)
    # reportObj.move_report_files()
    # run_dict = {'vanilla_tree': {'run': True, 'mode': 'train','name': 'vanilla_tree', 'params':{'random_state': 10} },

    run_dict['vanilla_tree']['run']
    # print(run_dict['vanilla_tree']['params']['random_state'])
    vis = dv()

    # print(parser_red_wine.y.head())
    dataset_comment = "redwine ds, Y <6 == 0, test size =0.3"
    multi = data_wrangling['multi'];
    report_list = []
    test_list = []
    # SECTION 1: Evaluate Decission Trees
    model_list = {}

    if run_dict['vanilla_tree']['run'] or run_dict['grid_tree']['run']:
        # 1.1. Test vanilla tree
        # name1 = 'vanilla_tree'
        params = {'random_state': 10}
        dt = DecisionTreeClassifier(**params)
        rprt, vanilla_tree = evaluate_algo(dt, params, parser_mushroom,
                                           run_dict['vanilla_tree']['name'], multi=multi)
        report_list.append({**rprt, **data_wrangling})
        vanilla_tree.set_report({**rprt, **data_wrangling})
        model_list[run_dict['vanilla_tree']['name']] = vanilla_tree
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['vanilla_tree']['name'], vanilla_tree)
    elif run_dict['vanilla_tree']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['vanilla_tree']['pickle'])
        print(path)
        vanilla_tree = pickle.load(open(path, 'rb'))
        if run_dict['vanilla_tree']['test']:
            test_report_list.append(test_model(vanilla_tree, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(vanilla_tree.get_report())
        # 1.1 END

    if (run_dict['grid_tree']['run']):
        # 1.2. Train tree with best grid search params
        # Get hyper parameters for grid search
        path = vanilla_tree.estimator.cost_complexity_pruning_path(parser_mushroom.XTrain, parser_mushroom.yTrain)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        param_grid = [
            {'criterion': ["gini", "entropy"], 'ccp_alpha': ccp_alphas[:-1], 'splitter': ["best", "random"],
             'random_state': [10]}
        ]
        # param_grid = [
        #         {'criterion': ["gini"], 'max_depth': [2], 'splitter': ["random"],
        #          'random_state': [10]}
        #     ]

        params, grid_time, grid_search_tree = general_grid_search(DecisionTreeClassifier(), param_grid,
                                                                  parser_mushroom.XTrain, parser_mushroom.yTrain)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree']['name'] + '_SEARCH',
                        grid_search_tree)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree']['name'] + '_SEARCH_TIME',
                        grid_time)
        # name2 = 'grid_search_tree'
        rprt, grid_tree = evaluate_algo(DecisionTreeClassifier(**params), params, parser_mushroom,
                                        run_dict['grid_tree']['name'], multi=multi)
        model_list[run_dict['grid_tree']['name']] = run_dict['grid_tree']['name']
        report_list.append(rprt)
        grid_tree.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree']['name'], grid_tree)
    elif run_dict['grid_tree']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_tree']['pickle'])
        print(path)
        grid_tree = pickle.load(open(path, 'rb'))
        if run_dict['grid_tree']['test']:
            test_report_list.append(test_model(grid_tree, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(grid_tree.get_report())
    # SECTION 2: Evaluate KNN
    # if (run_dict['grid_tree2']['run']):
    #
    #     param_grid = [
    #         {'criterion': ["gini", "entropy"], 'max_depth': [2,4,8,16,32,64,128], 'splitter': ["random"],
    #          'random_state': [10]}
    #     ]
    #     params, grid_time, grid_search_tree = general_grid_search(DecisionTreeClassifier(), param_grid,
    #                                                               parser_mushroom.XTrain, parser_mushroom.yTrain)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree2']['name'] + '_SEARCH',
    #                     grid_search_tree)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree2']['name'] + '_SEARCH_TIME',
    #                     grid_time)
    #     rprt, grid_tree2 = evaluate_algo(DecisionTreeClassifier(), params, parser_mushroom,
    #                                     run_dict['grid_tree2']['name'], multi=multi)
    #     model_list[run_dict['grid_tree2']['name']] = run_dict['grid_tree2']['name']
    #     report_list.append(rprt)
    #     grid_tree2.set_report(rprt)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_tree2']['name'], grid_tree2)
    # elif run_dict['grid_tree2']['pickle'] is not None:
    #     path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_tree2']['pickle'])
    #     print(path)
    #     grid_tree2 = pickle.load(open(path, 'rb'))
    #     if run_dict['grid_tree2']['test']:
    #         test_report_list.append(test_model(grid_tree2, parser_mushroom.XTest,  parser_mushroom.yTest))
    #     report_list.append(grid_tree2.get_report())
    if (run_dict['vanilla_knn']['run']):
        # 2.1 Vanilla KNN
        # name3 = 'vanilla_knn'
        params = {'random_state': 10}
        rprt, vanilla_knn = evaluate_algo(KNeighborsClassifier(), params, parser_mushroom,
                                          run_dict['vanilla_knn']['name'], multi=multi)
        model_list[run_dict['vanilla_knn']['name']] = vanilla_knn
        report_list.append(rprt)
        vanilla_knn.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['vanilla_knn']['name'], vanilla_knn)
    elif run_dict['vanilla_knn']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['vanilla_knn']['pickle'])
        print(path)
        vanilla_knn = pickle.load(open(path, 'rb'))
        if run_dict['vanilla_knn']['test']:
            test_report_list.append(test_model(vanilla_knn, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(vanilla_knn.get_report())

    if (run_dict['grid_knn']['run']):  # 2.2 Grid Search KNN
        # name4 = 'grid_knn'
        if multi == False:
            param_grid = [
                {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'p': [1, 2],
                 'algorithm': ['ball_tree', 'kd_tree'], 'leaf_size': [2, 10, 20]}
            ]
        else:
            # need to deal with one hot encoding
            param_grid = [
                {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}]

        params, grid_time, grid_search_knn = general_grid_search(KNeighborsClassifier(), param_grid,
                                                                 parser_mushroom.XTrain, parser_mushroom.yTrain)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_knn']['name'] + '_SEARCH',
                        grid_search_knn)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_knn']['name'] + '_SEARCH_TIME',
                        grid_time)
        rprt, grid_knn = evaluate_algo(KNeighborsClassifier(**params), params, parser_mushroom, run_dict['grid_knn']['name'],
                                       multi=multi)
        model_list[run_dict['grid_knn']['name']] = grid_knn
        report_list.append(rprt)
        grid_knn.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_knn']['name'], grid_knn)
    elif run_dict['grid_knn']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_knn']['pickle'])
        print(path)
        grid_knn = pickle.load(open(path, 'rb'))
        if run_dict['grid_knn']['test']:
            test_report_list.append(test_model(grid_knn, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(grid_knn.get_report())

    if run_dict['vanilla_ann']['run']:

        # # 3 Neural Network
        #
        # 3.1 Vanilla ANN
        params = {'random_state': 10}
        rprt, vanilla_ann = evaluate_algo(MLPClassifier(max_iter=10000), params, parser_mushroom,
                                          run_dict['vanilla_ann']['name'], multi=multi)
        model_list[run_dict['vanilla_ann']['name']] = vanilla_ann
        report_list.append(rprt)
        vanilla_ann.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['vanilla_ann']['name'], vanilla_ann)
    elif run_dict['vanilla_ann']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['vanilla_ann']['pickle'])
        print(path)
        vanilla_ann = pickle.load(open(path, 'rb'))
        if run_dict['vanilla_ann']['test']:
            test_report_list.append(test_model(vanilla_ann, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(vanilla_ann.get_report())
    if run_dict['grid_ann']['run']:
        param_grid = [
            {'hidden_layer_sizes': [(6,), (12,), (24,), (6, 3), (12, 3), (12, 6)],
             'solver': ['sgd', 'adam'],
             'alpha': [.000001, .000020, .0001, .001],
             'learning_rate': ['invscaling'],
             'momentum': [0, .02, .1],
             'early_stopping': [True], }
        ]

        params, grid_time, grid_search_ann = general_grid_search(MLPClassifier(max_iter=10000), param_grid,
                                                                 parser_mushroom.XTrain, parser_mushroom.yTrain)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_ann']['name'] + '_SEARCH',
                        grid_search_ann)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_ann']['name'] + '_SEARCH_TIME',
                        grid_time)
        print(run_dict['grid_ann']['name']+"params:", params)
        rprt, grid_ann = evaluate_algo(MLPClassifier(**params), params, parser_mushroom,
                                       run_dict['grid_ann']['name'], multi=multi)
        model_list[run_dict['grid_ann']['name']] = grid_ann
        report_list.append(rprt)
        grid_ann.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_ann']['name'], grid_ann)

    elif run_dict['grid_ann']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_ann']['pickle'])
        print(path)
        grid_ann = pickle.load(open(path, 'rb'))
        if run_dict['grid_ann']['test']:
            test_report_list.append(test_model(grid_ann, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(grid_ann.get_report())
    if run_dict['grid_boost']['run']:
        param_grid = [
            {'criterion': ['friedman_mse'],
             'n_estimators': [100, 200, 300],
             'subsample': [.2, .5, .8, 1],
             'tol': [.00005, .0001, .0002],
             'max_depth': [2, 4, 8, 16]}
        ]

        params, grid_time, grid_search_boosting = general_grid_search(GradientBoostingClassifier(), param_grid,
                                                                      parser_mushroom.XTrain, parser_mushroom.yTrain)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_boost']['name'] + '_SEARCH',
                        grid_search_boosting)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_boost']['name'] + '_SEARCH_TIME',
                        grid_time)
        print("best params:", params)
        rprt, grid_boosting = evaluate_algo(GradientBoostingClassifier(**params), params, parser_mushroom,
                                            run_dict['grid_boost']['name'], multi=multi)
        model_list[run_dict['grid_boost']['name']] = grid_boosting
        report_list.append(rprt)
        grid_boosting.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_boost']['name'], grid_boosting)

    elif run_dict['grid_boost']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_boost']['pickle'])
        print(path)
        grid_boosting = pickle.load(open(path, 'rb'))
        if run_dict['grid_boost']['test']:
            test_report_list.append(test_model(grid_boosting, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(grid_boosting.get_report())
    # if run_dict['grid_svm']['run']:
    #     param_grid = [
    #         {'kernel': ["linear", "poly", "rbf", "sigmoid"], 'shrinking': [True, False], 'probability': [True, False]}
    #     ]
    #     params, grid_time, grid_search_svc = general_grid_search(SVC(), param_grid, parser_mushroom.XTrain,
    #                                                              parser_mushroom.yTrain)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm']['name'] + '_SEARCH',
    #                     grid_search_svc)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm']['name'] + '_SEARCH_TIME',
    #                     grid_time)
    #     print("best params:", params)
    #     rprt, grid_svc = evaluate_algo(GradientBoostingClassifier(**params), params, parser_mushroom,
    #                                    run_dict['grid_svm']['name'], multi=multi)
    #     model_list[run_dict['grid_svm']['name']] = grid_svc
    #     report_list.append(rprt)
    #     grid_svc.set_report(rprt)
    #     pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm']['name'], grid_svc)
    # elif run_dict['grid_svm']['pickle'] is not None:
    #     path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_svm']['pickle'])
    #     print(path)
    #     grid_svc = pickle.load(open(path, 'rb'))
    #     if run_dict['grid_svm']['test']:
    #         test_report_list.append(test_model(grid_svc, parser_mushroom.XTest,  parser_mushroom.yTest))
    #     report_list.append(grid_svc.get_report())

    if run_dict['grid_svm2']['run']:
        param_grid = [{'kernel': ["rbf"],
                       'gamma': [.1, .5, 1, 2, 4],
                       'C': [.001, .01, .1, 1, 10, 100, 1000], }]
        params, grid_time, grid_search_svc2 = general_grid_search(SVC(), param_grid, parser_mushroom.XTrain,
                                                                  parser_mushroom.yTrain)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm2']['name'] + '_SEARCH',
                        grid_search_svc2)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm2']['name'] + '_SEARCH_TIME',
                        grid_time)
        print("best params:", params)
        rprt, grid_svc2 = evaluate_algo(SVC(**params), params, parser_mushroom,
                                        run_dict['grid_svm2']['name'], multi=multi)
        model_list[run_dict['grid_svm']['name']] = grid_svc2
        report_list.append(rprt)
        grid_svc2.set_report(rprt)
        pickle_and_move(data_wrangling['data_set_name'] + '_' + run_dict['grid_svm2']['name'], grid_svc2)
    elif run_dict['grid_svm2']['pickle'] is not None:
        path = os.path.join(os.getcwd(), 'input', run_dict['parser']['type'], run_dict['grid_svm2']['pickle'])
        print(path)
        grid_svc2 = pickle.load(open(path, 'rb'))
        if run_dict['grid_svm2']['test']:
            test_report_list.append(test_model(grid_svc2, parser_mushroom.XTest,  parser_mushroom.yTest))
        report_list.append(grid_svc2.get_report())
    # name9 = 'vanilla_svc'
    # params = {}
    # rprt, grid_svc = evaluate_algo(GradientBoostingClassifier(), params, parser_mushroom, name9, multi=multi)
    # model_list[name9] = grid_svc
    # report_list.append(rprt)

    # ALWAYS AT THE END
    df_report_test = pd.DataFrame.from_dict(test_report_list, orient='columns')
    df_report = pd.DataFrame.from_dict(report_list, orient='columns')
    print(df_report)
    df_report.to_csv(r'test_report.csv')
    df_report_test.to_csv(r'test_report_on_testdata.csv')
    print("You reached the end")
    file_name = 'models.pickle '
    outfile = open(file_name, 'wb')
    pickle.dump(model_list, outfile)
    outfile.close()

    # file_name = 'parser.pickle '
    # outfile = open(file_name,'wb')
    # pickle.dump(parser_mushroom, outfile)
    # outfile.close()
    reportObj.move_report_files()
    # name1 = 'vanilla_tree'
    # params = {'random_state': 10}
    # rprt, vanilla_tree = evaluate_algo(DecisionTreeClassifier(**params), params, parser_red_wine, name1)
    # report_list.append(rprt)
    # model_list[name1] = vanilla_tree