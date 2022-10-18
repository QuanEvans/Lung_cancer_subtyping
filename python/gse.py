import GEOparse
import pandas as pd
import random as rd
from sklearn.cluster import KMeans
import itertools

class GSEs:
    
    def __init__(self,*,file_path:str=None,gse:GEOparse=None):
        """initialize the class

        Args:
            file_path (str, optional): path of soft file. Defaults to None.
            gse (GEOparse.GSE, optional): read gse directly if gse object is provided else use read_soft. Defaults to None.
            matrix (array, optional): gene expression matrix. Defaults to None.
            annotations (dataframe, optional): disease type annotations of the matrix. Defaults to None.
            model (kmeans.Model, optional): kmeans model. Defaults to None.
            training_data (dataframe, optional): training data. Defaults to None.
            testing_data (dataframe, optional): testing data. Defaults to None.
            predict_matrix (dataframe, optional): predictions cluster number dataframe. Defaults to None.
            accuracy_matrix (dataframe, optional): accuracy dataframe. Defaults to None.
            accuracy (float, optional): accuracy of the model. Defaults to None.
        """
        
        self.gse = gse #GEOparse.GSE object
        self.seed = None #random seed
        self.matrix = None #pandas dataframe
        self.annotations = None #dataframe
        self.model = None #kmeans model
        self.training_data = None #dataframe
        self.testing_data = None #dataframe
        self.predict_matrix = None #predictions cluster number dataframe
        self.accuracy_matrix = None #accuracy dataframe
        self.accuracy = None #prediction accuracy in float

        if file_path:
            self.file_path = file_path
            self.read_soft(file_path=file_path) #read soft file
        if gse is not None:
            self._initialize()
    
    def _initialize(self):
        """
        initialize the class get the gene expression matrix and annotations
        """
        self._get_matrix()._get_annotations()
        return self


    def read_soft(self, file_path):
        self.gse = GEOparse.get_GEO(filepath=file_path) 
        self._initialize() #initialize the parameters for later use
        return self
    
    def _get_matrix(self):
        """get the gene expression matrix"""
        if self.gse:
            self.matrix = self.gse.pivot_samples('VALUE').T
        else:
            raise ReferenceError('No file path')
        return self

    def _get_annotations(self):
        """table is a pandas dataframe with the following columns: sample_id and characteristics_ch1.0.disease state

        Returns:
            self: for a chain call
        """
        self.annotations = self.gse.phenotype_data[['characteristics_ch1.0.disease state']].copy()
        return self
    
    def _subset(self, fraction:float, matrix=None):
        """subset the matrix into training and testing data with fraction.
        Each disease subset will have the same proportion of training and testing data

        Args:
            fraction (float): fraction of training data
            matrix (_type_, optional): gene expression matrix. Defaults to None and will use self.matrix.

        Raises:
            ValueError: fraction must be between 0 and 1

        Returns:
            self: for a chain call
        """
        if matrix is None:
            matrix = self.matrix.copy()
        elif fraction<0 or fraction>1:
            raise ValueError('fraction must be between 0 and 1')
        merged_dataframe = self.annotations.merge(matrix, left_index=True, right_index=True) #add disease state to the gene expression matrix
        training_data = merged_dataframe.groupby('characteristics_ch1.0.disease state').apply(lambda x: x.sample(random_state=self.seed,frac=fraction)).droplevel(0) #training data with balanced disease type
        testing_data = merged_dataframe.drop(training_data.index) # drop the training data from the original dataframe to get the testing data
        # sort by id
        training_data = training_data.sort_index()
        testing_data = testing_data.sort_index()

        self.training_data = training_data
        self.testing_data = testing_data

        return self

    
    def train_model(self, n_clusters=2,train_frac: float=1.0, matrix=None, seed=None):
        """using kmeans to train model

        Args:
            n_clusters (int, optional): number of k means cluster. Defaults to 2.
            matrix (array, optional): matrix with gene expression information. Defaults to None.
            train_size (float, optional): precentage of training sample. Defaults to None.
            seed (int, optional): random seed. Defaults to None.
        Returns:
            self: return self for a chain call
        """
        self._reset_model() #reset the model first
        
        if seed:#set random seed if provided
            seed = seed
        else:
            seed = self.seed

        if matrix is None:
            self._get_matrix()
            matrix = self.matrix

        if train_frac !=1.0:
            self._subset(train_frac, matrix)
            matrix = self.training_data

        if 'characteristics_ch1.0.disease state' in matrix.columns: #drop the disease state column from the matrix
            matrix = matrix.drop('characteristics_ch1.0.disease state', axis=1) 

        self.model = KMeans(n_clusters=n_clusters, random_state=seed).fit(matrix)

        return self

    def predict(self, matrix=None, all_data: bool=False):
        """predict the cluster number for the testing data

        Args:
            matrix (dataframe, optional): gene expression matrix. Defaults to None.
            all_data (bool, optional): Used for fracton test_data only, TRUE will test on all_data. Defaults to False test on test_data.
            
        Returns:
            self: return self for a chain call
        """
        if matrix is None:
            matrix = self.matrix
        if all_data is False: #if all_data is false, use the testing data
            if self.testing_data is not None: #if testing data is not none, use the testing data
                matrix = self.testing_data
                if 'characteristics_ch1.0.disease state' in matrix.columns: #drop the disease state column from the matrix
                    matrix = matrix.drop('characteristics_ch1.0.disease state', axis=1) 

        predictions = self.model.predict(matrix)
        predictions = pd.DataFrame(predictions, index=matrix.index, columns=['cluster_number'])
        self.predict_matrix = predictions

        self._accuracy_data()

        return self

    def _accuracy_data(self):
        """test the accuracy of the model.
        Since kmeans is unsupervised learning, the cluster might not be the same as the disease type.
        This function will use premutation to find the best match between the cluster and the disease type
        pravite function. Use predict() to get the accuracy

        Raises:
            ReferenceError: No predictions data

        Returns:
            self: return self for a chain call
        """

        if self.predict_matrix is None:
            raise ReferenceError('No predictions')
        accuracy_df = self.predict_matrix.merge(self.annotations, left_index=True, right_index=True)
        
        lables = list(self.annotations['characteristics_ch1.0.disease state'].unique()) # get the unique disease type

        max_accuracy = 0
        max_accuracy_df = None
        for permutation in itertools.permutations(lables): #permutation of the disease type
            num_key = list(range(len(lables)))
            lables_dict = dict(zip(num_key, permutation))
            accuracy_df['prediction'] = accuracy_df['cluster_number'].replace(lables_dict)
            cur_accuracy = accuracy_df[accuracy_df['characteristics_ch1.0.disease state'] == accuracy_df['prediction']].shape[0]/accuracy_df.shape[0] #calculate the accuracy

            if cur_accuracy > max_accuracy: #if the current accuracy is higher than the max accuracy, update the max accuracy
                max_accuracy = cur_accuracy
                max_accuracy_df = accuracy_df.copy()
    
        self.accuracy_matrix = max_accuracy_df
        self.accuracy = max_accuracy
        return self

    def set_seed(self, seed):
        self.seed = seed
        return self

    def get_aaccuracy_matrix(self):
        return self.accuracy_matrix
    
    def get_accuracy(self):
        return self.accuracy
    
    def _reset_model(self):
        self.model = None #kmeans model
        self.training_data = None #dataframe
        self.testing_data = None #dataframe
        self.predict_matrix = None #predictions cluster number dataframe
        self.accuracy_matrix = None #accuracy dataframe
        self.accuracy = None #prediction accuracy in float
        return self




    

    
