# Building an end to end Full Pipeline for Machine Learning
import os
import numpy as np
import pandas as pd
from typing import Any, Optional
from warnings import simplefilter


# CompletePipeline class
class CompletePipeline(object):
    """This CustomPipeline class"""
    
    """
    [CompletePipeline docs]:
    CustomPipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `fit` and `transform` methods.
    The final estimator only needs to implement `fit`.
    
    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    'passthrough' or 'None'.

    Read more in the :ref: User Guide CompletePipeline.
    Done By Rohan

    versionadded: 1.0

    Parameters
    ----------
    **kwargs : dictionary
        List of model steps as keys and  values of the acutal model's (implementing `fit`/`transform`) that
        are chained, in the order in which they are chained, with the last
        object an estimator.

    See Also
    --------
    CompletePipeline which is a subclass of BasePipeline

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from custom_pipelines import CompletePipeline
    >>> X, y = make_classification(random_state=0)
    
    ...                                                    
    >>> params = {"data": df, "split_model": train_test_split(X,y, random_state=0), "transformer": StandardScaler(), model": 
    RandomForestClassifier()}
    >>> pipe = CompletePipeline(**params)
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train)
    CompletePipeline({'data': <class 'pandas.core.frame.DataFrame'>, 'split_model': <class 'list'>, 'model': <class'sklearn.ensemble._forest.RandomForestClassifier'>})
    >>> c_table, cr, cm = pipe.evaluate(X_test, y_test)
    """
    
    """
    WARNING:- To use custom_pipelines.CompletePipeline.dockerize() and 
    custom_pipelines.CompletePipeline.run_docker() You need to haved dokcer 
    installed in your local, virtual or cloud instance machine.
    """
                               
    def __init__(self, **kwargs):
        simplefilter("ignore")
        """
         Valid parameter keys can be listed with get_params(). Note that
        you can directly set the parameters of the estimators contained in
        `steps`.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in sequence.

        Returns
        -------
        self : object
            CompletePipeline class instance.
        """
        self.memory = kwargs.copy()
        self.shelf = {key: type(val) for key, val in kwargs.items()}
    
    def load_data(self):
        """Method Name: load_data
           Description: The method gives the DataFrame as output
           Parameters: Takes the target feature as an input
           Return Type: pd.DataFrame
        """
        try:
            self.data = self.memory["data"]
            return self.data
        except Exception as e:
            print("Error occured inside the load_data method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def impute(self):
        """Method Name: impute
           Description: The method takes the data as input and fit's the data with imputer
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.imputer = self.memory["imputer"]
            print("Filling the missing values...")
            self.imputed_data = self.imputer.fit_transform(self.data)
            return self.imputed_data
        except Exception as e:
            print("Error occured inside the impute method from the CompletePipeline class " + str(e))
            raise Exception()
    
    
    def detect(self, is_nan: bool=False):
        """Method Name: detect
           Description: The method takes the data as input and detects the outlier's present in the data
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.detector = self.memory["detector"]
            if is_nan is False:
                print("Detecting the outliers in the data...")
                self.detector.fit(self.data)
                self.detected_data = self.detector.predict(self.data)
                return self.detected_data
            else:
                print("Detecting the outliers in the imputed data...")
                self.detector.fit(self.imputed_data)
                self.detected_data = self.detector.predict(self.imputed_data)
                return self.detected_data
        except Exception as e:
            print("Error occured inside the detect method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def encode(self):
        """Method Name: encode
           Description: The method takes the data as input and gives the encoded DataFrame as output
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.encoder = self.memory["encoder"]
            print("Encoding the data...")
            self.encoded_data = self.encoder.fit_transform(self.data)
            return self.encoded_data
        except Exception as e:
            print("Error occured inside the encode method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def reverse_encode(self):
        """This is reverse_encode method"""
        try:
            labels = self.encoder.inverse_transform(self.encoded_data)
            return labels
        except Exception as e:
            print("Error occured inside the reverse_encode method from the CompletePipeline class " + str(e))
            raise Exception()
        
    def transform(self, numerical_features: Optional[list]):
        """Method Name: transform
           Description: The method takes the data as input and gives the DataFrame as output
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.transformer = self.memory["transformer"]
            print("Transforming the data...")
            self.data = self.load_data()
            self.data[numerical_features] = self.transformer.fit_transform(self.data[numerical_features])
            return self.data
        except Exception as e:
            print("Error occured inside the transform method from the CompletePipeline class " + str(e))
            raise Exception()
             
    def decompose(self):
        """Method Name: decompose
           Description: The method takes the data as input gives the reduced DataFrame as output
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.decomposer = self.memory["decomposer"]
            print("Decomposing the data...")
            self.reduced_data = self.decomposer.fit_transform(self.transformed_data)
            return self.reduced_data
        except Exception as e:
            print("Error occured inside the decompose method from the CompletePipeline class " + str(e))
            raise Exception()
            
    
    def cluster(self):
        """Method Name: cluster
           Description: The method takes the DataFrame as input gives the  clustered DataFrame as output
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.clusterer = self.memory["clusterer"]
            print("Clsutering the data...")
            self.clustered_data = self.clusterer.fit_transform(self.X_train)
            return self.clustered_data
        except Exception as e:
            print("Error occured inside the cluster methd from the CompletePipeline class " + str(e))
            raise Exception()
    
    def preprocess(self):
        """Method Name: preprocess
           Description: The method takes the DataFrame as input gives the  splitted DataFrame as output
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.split_model = self.memory["split_model"]
            print("Splitting the data...")
            self.X_train, self.X_test, self.y_train, self.y_test = self.split_model
        except Exception as e:
            print("Error occured inside the preprocess method from the CompletePipeline class" + str(e))
            raise Exception() 
            
    def cross_validate(self, X: pd.DataFrame):
        """Method Name: cross_validate
           Description: The method applies cross validation on the given DataFrame input
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.cv_model = self.memory["cv_model"]
            print("Cross Validating the data...")
            for train_idx, test_idex in self.cv_model.split(X):
                self.X_train, self.X_test = X[train_idx], X[test_idx]
                self.y_train, self.y_test = y[train_idx], y[test_idx]
            return self.X_train, self.y_train
        except Exception as e:
            print("Error occured inside the cross_validate method from the CompletePipeline class" + str(e))
            raise Exception()
            
    def get_best_metrics(self):
        """Method Name: get_hyper_tune_metrics
           Decription: The method get_hyper_tune_metrics gives the best parameters of the model and the accuracy score of the model
           Parametrs: None
           Return Type: best_paramters, float
        """
        try:
            if self.tune == True:
                return (self.tuned_model.best_params_, self.tuned_model.best_score_)
            else:
                return None
        except Exception as e:
            print("Error occured inside the get_hyper_tune_metrics method from the CompletePipeline class " + str(e))
            raise Exception() 
                      
    def fit(self, tune: bool=False):
        """Method Name: fit
           Description: The method fit's the ML model with the input DataFrame
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.tune = tune
            if self.tune == True:
                self.tuned_model = self.memory["hyper_tune_model"]
                if self.y_train.values.ndim > 1:
                    print("Flattening the target variable and Fitting the model...")
                    print("Applying Hyperparameter tuning...")
                    self.tuned_model.fit(self.X_train.values, self.y_train.values.ravel())
                else:
                    print("Fitting the model...")
                    print("Applying Hyperparameter tuning...")
                    return self.tuned_model.fit(self.X_train.values, self.y_train.values)
            else:
                self.model = self.memory["model"]
                print("Fitting the model...")
                return self.model.fit(self.X_train.values, self.y_train.values)
        except Exception as e:
            print("Error occured inside the fit method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def predict(self):
        """Method Name: predict
           Description: The method gives the prediction results of the input DataFrame
           Parameters: None
           Return Type: np.ndarray
        """
        try:
            if self.tune == True:
                print("Predicting the target data after hyperparameter tuning...")
                return self.tuned_model.predict(self.X_test.values)
            else:
                print("Predicting the target data...")
                return self.model.predict(self.X_test.values)
        except Exception as e:
            print("Error occured inside the predict method from the CompletePipeline class" + str(e))
            raise Exception()
    
    def predict_proba(self, tune: bool=False):
        """Method Name: predict_proba
           Description: The method gives the probablity scores as output
           Parameters: None
           Return Type: np.ndarray
        """
        try:
            if tune == True:
                print("Getting probability scores...")
                return self.tuned_model.predict_proba(self.X_test)
            else:
                print("Getting probability scores...")
                return self.model.predict_proba(self.X_test)
        except Exception as e:
            print("Error occured inside the predict_proba method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def evaluate(self, y_pred: np.ndarray, problem_type: str):
        """Method Name: evaluate
           Description: The method gives classification metrics if the 
                        model_type is classification else gives the regression metrics
           Parameters: None
           Return Type: Table
        """
        try:
            from prettytable import PrettyTable
            valid_types = ["classification", "regression"]
            if problem_type not in valid_types:
                raise ValueError(f"{problem_type} is not a valid problem type!")
            elif problem_type == "classification":
                print("Getting classification metrics...")
                from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
                from sklearn.metrics import classification_report, confusion_matrix
                clf_table = PrettyTable()
                clf_table.field_names = ["Precision", "Accuracy", "Recall", "F1_Score"]
                clf_table.add_row([
                                      round(precision_score(self.y_test, y_pred, average="micro"), 3),
                                      round(accuracy_score(self.y_test, y_pred), 3),
                                      round(recall_score(self.y_test, y_pred, average="micro"), 3),
                                      round(f1_score(self.y_test, y_pred, average="micro"), 3)
                                  ])
                self.cm = confusion_matrix(self.y_test, y_pred)
                return (
                            clf_table, 
                            classification_report(self.y_test, y_pred), 
                            confusion_matrix(self.y_test, y_pred)
                       )
            elif problem_type == "regression":
                print("Getting regression metrics...")
                from sklearn.metrics import mean_squared_error, r2_score
                import numpy as np
                reg_table = PrettyTable()
                reg_table.field_names = ["Mean_Squared_Error", "Root_Mean_Squared_Error", "R2_Score"]
                reg_table.add_row([
                                        round(mean_squared_error(self.y_test, y_pred), 3),
                                        round(np.sqrt(mean_squared_error(self.y_test, y_pred)), 3),
                                        round(r2_score(self.y_test, y_pred), 3)
                                  ])
                return reg_table
            else:
                return "This step will not be executed!"
            
        except Exception as e:
            print("Error occured inside the evaluate method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def plot_confusion_matrix(self):
        """Method Name: plot_confusion_matrix
           Description: The method plot's the confusion matrix in jupyter notebook
           Parameters: None
           Return Type: matplotlib.axes._subplots.AxesSubplot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.title("Confusion Matrix")
            return sns.heatmap(self.cm, annot=True)
        except Exception as e:
            print("Error occured inside the plot_confusion_matrix method from the CompletePipeline class " + str(e))
            raise Exception()
    
    
    def get_gain_chart(self, df: pd.DataFrame, actual_col: str, predicted_col: str, probability_col: str):
        """Method Name: get_gain_chart
           Description: The method get's the gain chart
           Parameters: pd.DataFrame, str, str, str
           Return Type: pd.DataFrame
        """
        try:
            from sklearn.metrics import accuracy_score
            df.sort_values(by=probability_col, ascending=False, inplace=True)
            subset = df[df[predicted_col] == True]
            rows = []
            for group in np.array_split(subset, 10):
                score = accuracy_score(group[actual_col].tolist(),
                                                           group[predicted_col].tolist(),
                                                           normalize=False)
                rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})
            gain = pd.DataFrame(rows)
            # Cumulative Gains Calculation
            gain["RunningCorrect"] = gain["NumCorrectPredictions"].cumsum()
            gain["PercentCorrect"] = gain.apply(
                lambda x: (100 / gain["NumCorrectPredictions"].sum()) * x["RunningCorrect"], axis=1)
            gain["CumulativeCorrectBestCase"] = gain["NumCases"].cumsum()
            gain["PercentCorrectBestCase"] = gain["CumulativeCorrectBestCase"].apply(
                lambda x: 100 if (100 / gain["NumCorrectPredictions"].sum()) * x > 100 else (100 / gain["NumCorrectPredictions"].sum()) * x)
            gain["AvgCase"] = gain["NumCorrectPredictions"].sum() / len(gain)
            gain["CumulativeAvgCase"] = gain["AvgCase"].cumsum()
            gain["PercentAvgCase"] = gain["CumulativeAvgCase"].apply(
                lambda x: (100 / gain["NumCorrectPredictions"].sum()) * x)

            #Gain Chart
            gain["NormalisedPercentAvg"] = 1
            gain["NormalisedPercentWithModel"] = gain["PercentCorrect"] / gain["PercentAvgCase"]
            gain.insert(0, "Decile", gain.index + 1)
            return gain
        except Exception as e:
            print("Error occured inside the get_gain_chart method from the CompletePipeline class " + str(e))
            raise Exception()
        
    
    
    def plot_gain_chart(self, gain: pd.DataFrame):
        """Method Name: plot_gain_chart
           Description: The method plot's the gain chart in jupyter notebook
           Parameters: pd.DataFrame
           Return Type: matplotlib.axes._subplots.AxesSubplot
        """
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            fig.canvas.draw()

            handles = []
            handles.append(ax.plot(gain["PercentCorrect"], "r-", label="Percent Correct Predictions"))
            handles.append(ax.plot(gain["PercentCorrectBestCase"], "g-", label="Best Case (for current model)"))
            handles.append(ax.plot(gain["PercentAvgCase"], "b-", label="Average Case (for current model)"))
            ax.set_xlabel('Total Population (%)')
            ax.set_ylabel('Number of Respondents (%)')

            ax.set_xlim([0, 9])
            ax.set_ylim([10, 100])

            labels = [int((label+1)*10) for label in [float(item.get_text()) for item in ax.get_xticklabels()]]

            ax.set_xticklabels(labels)

            fig.legend(handles, labels=[h[0].get_label() for h in handles])
            fig.show()
        except Exception as e:
            print("Error occured inside the plot_gain_chart method from the CompletePipeline class " + str(e))
            raise Exception()
        
    
    def resample(self, features: pd.DataFrame, target: pd.DataFrame):
        """Method Name: resample
           Description: The method resamples (Undersample or Oversampling) the train data
           Parameters: None
           Return Type: pd.DataFrame
        """
        try:
            self.sampler = self.memory["sampler"]
            print("Applying sampling to the training dataset...")
            self.X_train, self.y_train = self.sampler.fit_resample(features, target)
            return self.X_train, self.y_train
        except Exception as e:
            print("Error occured inside the resample method from the CompletePipeline class " + str(e))
            raise Exception() 
    
                   
    def save(self, filename: str):
        """Method Name: save
           Description: The method save's the trained model in given fileformat
           Parameters: None
           Return Type: list
        """
        try:
            from joblib import dump
            print("Saving the model...")
            if self.tune == True:
                return dump(self.tuned_model, filename)
            else:
                return dump(self.model, filename)
        except Exception as e:
            print("Error occured inside the save method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def load(self, filename: str):
        """Method Name: load
           Description: The method load's the trained model in given fileformat
           Parameters: None
           Return Type: sklearn
        """
        try:
            from joblib import load
            print("Loading the model...")
            self.loaded_model = load(filename)
            return self.loaded_model
        except Exception as e:
            print("Error occured inside the save method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def dockerize(self, container_name: str, model_path: str, dir_path: str, file_name: str):
        """Method Name: dockerize
           Description: The method dockerize's the trained model in given fileformat
           Parameters: None
           Return Type: None
        """
        try:
            simplefilter("ignore")
            print("Checking for docker...")
            flag = os.system("docker --version")
            if flag != 0:
                print("There is no docker present in your machine!")
                print(f"Exitted with flag {flag}!")
            self.cn = container_name
            dockerfile = os.path.isfile("Dockerfile")
            if dockerfile is True:
                cmd = ["FROM python:3.8 \n\n", 
                       "WORKDIR /modeldir \n\n", 
                       "COPY requirements.txt requirements.txt \n\n",
                       "RUN pip install -r requirements.txt \n\n",
                       "ADD . . \n\n",
                       f"COPY {model_path} ./modeldir \n\n",
                       f"""CMD ["python", {file_name}]"""]
                with open("Dockerfile", "w") as f:
                    f.writelines(cmd)
                print("Dockerizing the CompletePipeline...")
                os.system(f"docker build  -f {dir_path}/Dockerfile -t {container_name} . ")
            else:
                PATH = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/datascience-ml-dev3/code/Users/datascience-aml/Rohan/pipelines"
                print("Dockerizing the CompletePipeline...")
                os.system(f"docker build  -f {PATH}/Dockerfile -t {container_name} . ")
                print("Completed building the docker image!")
        except Exception as e:
            print("Error occured: inside the dockerize method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def run_docker(self):
        """Method Name: run_docker
           Description: The method run's the dockerized Pipeline from the CompletePipeline class
           Parameters: None
           Return Type: None
        """
        try:
            print("Running the created docker image...")
            os.system(f"docker run -it {self.cn}")
            print("Successfully ran the docker image!")
        except Exception as e:
            print("Error occured inside the run_docker method from the CompletePipeline class " + str(e))
            raise Exception()
    
    def get_method_names(self):
        """Method Name: get_method_names
           Description: The method give's all the method name's of the CompletePipeline class
           Parameters: None
           Return Type: list
        """
        try:
            method_names = ["load_data", "impute", "detect", "encode", "reverse_encode", 
                            "transform", "decompose", "cluster", "preprocess", "cross_validate", 
                            "fit", "predict", "predict_proba", "evaluate", "resample",
                           "plot_confusion_matrix", "save", "load", "dockerize",
                            "run_docker", "get_best_metrics", "plot_gain_chart"
                           "get_chain_chart"]
            return method_names
        except Exception as e:
            print("Error occured inside the get_method_names method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def __add__(self, other):
        """This special method add's the two CompletePipeline objetcs and gives the resulting CompletePipeline object"""
        try:
            self.bag = {**self.memory, **other.memory}
            return CompletePipeline(**self.bag)
        except Exception as e:
            print("Error occured inside the __add__ method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def __iter__(self):
        """This special method allows to iterate over the CompletePipeline object"""
        try:
            for i in self.memory.values():
                yield i
        except Exception as e:
            print("Error occured inside the __iter__ method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def __len__(self):
        """This special  method gives the components present in the CompletePipeline"""
        try:
            return len(self.shelf)
        except Exception as e:
            print("Error occured inside the __len__ method from the CompletePipeline class " + str(e))
            raise Exception()
            
    def __repr__(self):
        """This is special method to represent the CompletePipeline object onto terminal"""
        try:
            return f"CompletePipeline({list(self.shelf.items())})"
        except Exception as e:
            print("Error occured inside the __repr__ method from the CompletePipeline class " + str(e))
            raise Exception()
