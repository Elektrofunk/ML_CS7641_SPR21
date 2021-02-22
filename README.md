
# ML_CS7641_SPR21
Git Repository

# How to install

using anaconda create a new environment python 3.6 environment
use the requirements.txt file found on the repository directory to create your envioronment.
pip install -r requirements.txt

# How to run

To replicate results watch this video tutorial
https://youtu.be/YIZYxAofG_Y

All the information below can be found on the video tutorial as well.

Remeber you can move the pickle files found inside ./input/wine/test_wine_boosting to ./input/wine/
Then in main.py you can set the title variable to wine. Then in the nested dictionary you can set it like this then run:


![Dictionary_Configuration](/img/DictionaryConfig.PNG)

![Folder_Struct](/img/FolderStructure.PNG)
Put the pickle files inside the input directory.
![Input_Dir](/img/InputDirectory.PNG)
Outputs will come out inside the directory which is created with a date.
![Output_Dir](/img/OutputDirectory.PNG)

If you want to retrain the models you can use the  wine_and_thyroid_summaries.xlx to create the models.
![Summaries_File](/img/Summaries.PNG)

note: Files only tested in Windows 10 (my linux box died in the middle of hw  :*())
Known issues: some older pickle files had an incorrect name (if you did mode.name)



# References
DataSetReferences can be found on the report. 

ml_tester.py

Code References:

If I am missing any reference it should be copied inside the code as a comment.

ml_tester.py

- Book: Hands on Machine Learning with Scikit-Learn, Keras and TensorFlow second edition. - Aurelien Geron


Report.py

- https: // stackoverflow.com / questions / 23556040 / moving - specific - file - types - with-python
- https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
- https: // datascience.stackexchange.com / questions / 27615 / should - we - apply - normalization - to - test - data -as-well

ParseDataSet.py        
- https://www.xspdf.com/help/50480465.html
- https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
- https://stackoverflow.com/questions/24645153/pandas-dataframe-columns-scaling-with-sklearn
- https: // datascience.stackexchange.com / questions / 27615 / should - we - apply - normalization - to - test - data -as-well

EstimatedEstimator.py
- source https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
-  https: // stackoverflow.com / questions / 45601897 / how - to - calculate - the - actual - size - of - a - fit - trained - model - in -sklearn
- https: // scikit - learn.org / stable / auto_examples / model_selection / plot_learning_curve.html
- https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
- https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
- https://plotly.com/python/roc-and-pr-curves/
- https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
- https://www.programcreek.com/python/example/81623/sklearn.metrics.classification_report

DatasetVis.py
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
- https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn
- https://www.programcreek.com/python/example/81623/sklearn.metrics.classification_report
- https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

cf_matrix.py
- https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
- https://github.com/DTrimarchi10/confusion_matrix/blob/master/cf_matrix.py
