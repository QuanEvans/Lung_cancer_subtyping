import GEOparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

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
            seed (int, optional): random state. Defaults to None.
            training_frac (float, optional): fraction of training data. Defaults to None.
            testing_frac (float, optional): fraction of testing data. Defaults to None.
            archive (dataframe, optional): archive of the predict result. Defaults to empty dataframe.
        """
        
        self.__gse = gse #GEOparse.GSE object
        self.__seed = None #random seed
        self.__matrix = None #pandas dataframe
        self.__annotations = None #dataframe
        self.__model = None #kmeans model
        self.__training_data = None #dataframe
        self.__training_frac = None #float
        self.__testing_data = None #dataframe
        self.__testing_frac = None #float
        self.__predict_matrix = None #predictions cluster number dataframe
        self.__accuracy_matrix = None #accuracy dataframe
        self.__accuracy = None #prediction accuracy in float
        self.__archive = pd.DataFrame() #archive of accuracy for different training size
        self.__pca = None #pca matrix

        if file_path:
            self.read_soft(file_path=file_path) #read soft file
        if gse is not None:
            self._initialize()

    #region property
    @property
    def gse(self):
        return self.__gse
    
    @property
    def seed(self):
        return self.__seed
    
    @property
    def matrix(self):
        return self.__matrix
    
    @property
    def annotations(self):
        return self.__annotations
    
    @property
    def model(self):
        return self.__model
    
    @property
    def training_data(self):
        return self.__training_data
    
    @property
    def testing_data(self):
        return self.__testing_data
    
    @property
    def predict_matrix(self):
        return self.__predict_matrix
    
    @property
    def accuracy_matrix(self):
        return self.__accuracy_matrix

    @property
    def accuracy(self):
        return self.__accuracy

    @property
    def archive(self):
        return self.__archive
    
    @property
    def pca_matrix(self):
        return self.__pca

    #endregion

    #region methods
    def _initialize(self):
        """
        initialize the class get the gene expression matrix and annotations
        """
        self._get_matrix()._get_annotations()
        return self


    def read_soft(self, file_path):
        self.__gse = GEOparse.get_GEO(filepath=file_path) 
        self._initialize() #initialize the parameters for later use
        return self
    
    def _get_matrix(self):
        """get the gene expression matrix"""
        if self.__gse:
            self.__matrix = self.__gse.pivot_samples('VALUE').T
        else:
            raise ReferenceError('No file path')
        return self

    def _get_annotations(self):
        """table is a pandas dataframe with the following columns: sample_id and characteristics_ch1.0.disease state

        Returns:
            self: for a chain call
        """
        self.__annotations = self.__gse.phenotype_data[['characteristics_ch1.0.disease state']].copy()
        return self
    
    def _subset(self, fraction:float, matrix=None):
        """subset the matrix into training and testing data with fraction.
        Each disease subset will have the same proportion of training and testing data

        Args:
            fraction (float): fraction of training data
            matrix (_type_, optional): gene expression matrix. Defaults to None and will use self.__matrix.

        Raises:
            ValueError: fraction must be between 0 and 1

        Returns:
            self: for a chain call
        """
        if matrix is None:
            matrix = self.__matrix.copy()
        elif fraction<=0 or fraction>1:
            raise ValueError('fraction must be between 0 and 1')
        merged_dataframe = self.__annotations.merge(matrix, left_index=True, right_index=True) #add disease state to the gene expression matrix
        training_data = merged_dataframe.groupby('characteristics_ch1.0.disease state').apply(lambda x: x.sample(random_state=self.__seed,frac=fraction)).droplevel(0) #training data with balanced disease type
        testing_data = merged_dataframe.drop(training_data.index) # drop the training data from the original dataframe to get the testing data
        # sort by id
        training_data = training_data.sort_index()
        testing_data = testing_data.sort_index()

        self.__training_data = training_data
        self.__testing_data = testing_data

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
        self.__training_frac = train_frac

        if seed:#set random seed if provided
            seed = seed
        else:
            seed = self.__seed

        if matrix is None:
            self._get_matrix()
            matrix = self.__matrix

        if train_frac !=1.0:
            self._subset(train_frac, matrix)
            matrix = self.__training_data

        if 'characteristics_ch1.0.disease state' in matrix.columns: #drop the disease state column from the matrix
            matrix = matrix.drop('characteristics_ch1.0.disease state', axis=1) 

        self.__model = KMeans(n_clusters=n_clusters, random_state=seed).fit(matrix)
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
            matrix = self.__matrix
        if all_data is False: #if all_data is false, use the testing data
            self.__testing_frac = 1-self.__training_frac
            if self.__testing_data is not None: #if testing data is not none, use the testing data
                matrix = self.__testing_data # if all_data is false, use the testing data
                if 'characteristics_ch1.0.disease state' in matrix.columns: #drop the disease state column from the matrix
                    matrix = matrix.drop('characteristics_ch1.0.disease state', axis=1) 
        else:
            self.__testing_frac = 1.0 #if all_data is true, use all data
            self.__testing_data = matrix

        predictions = self.__model.predict(matrix)
        predictions = pd.DataFrame(predictions, index=matrix.index, columns=['cluster_number'])
        self.__predict_matrix = predictions

        ## run pca 
        pca = self.pca(n_components=2,data=matrix)
        # archive data
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

        if self.__predict_matrix is None:
            raise ReferenceError('No predictions')
        accuracy_df = self.__predict_matrix.merge(self.__annotations, left_index=True, right_index=True)
        
        lables = list(self.__annotations['characteristics_ch1.0.disease state'].unique()) # get the unique disease type

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
    
        self.__accuracy_matrix = max_accuracy_df
        self.__accuracy = max_accuracy
        self._archive()
        return self

    def set_seed(self, seed):
        """set the random seed
        """
        self.__seed = seed
        return self

    def _archive(self):
        """archive the current accuracy to the archive dataframe
        """
        if self.__accuracy is not None:
            index = f'train_frac: {self.__training_frac}, test_frac: {self.__testing_frac}, seed: {self.__seed}'
            pca_df = pd.DataFrame()
            pca_df['type'] = self.__accuracy_matrix['characteristics_ch1.0.disease state']
            pca_df['PC1'] = self.__pca[:,0]
            pca_df['PC2'] = self.__pca[:,1]
            archive = pd.DataFrame({'training_frac': [self.__training_frac], 'testing_frac': [self.__testing_frac], \
                'accuracy': [self.__accuracy], 'seed':[self.__seed], 'pca':[pca_df],'accuracy_matrix':[self.__accuracy_matrix]}, index=[index])
            self.__archive = pd.concat([self.__archive, archive])
        return self

    def pca(self, n_components=2, data=None):
        """perform pca on the matrix
        """
        if data is None:
            data = self.__matrix
        pca = PCA(n_components=n_components)
        self.__pca = pca.fit_transform(data)
        return self
    
    def barplot_accuracy(self):
        """plot the accuracy of the model using seaborn barplot
        """
        if self.__archive is not None:
            archive = self.__archive.copy()
            sns.set(rc={'figure.figsize':(20.7,8.27)})
            sns.barplot(data=archive,x=archive.index, y='accuracy')
        return self
    
    def scatter_pca(self):
        pca = self.archive.pca
        pca_df = pd.DataFrame()
        indexs = pca.index
        for idx in indexs:
            cur_pca_df = pca.loc[idx].copy(deep=True)
            cur_pca_df['name'] = idx
            pca_df = pd.concat([pca_df, cur_pca_df], axis=0)
        sns.FacetGrid(pca_df, hue="type",col='name',height=6).map(plt.scatter, "PC1", "PC2").add_legend()
        return self

    def _reset_model(self):
        """reset the model and related porperties
        """
        self.__model = None #kmeans model
        self.__training_data = None #dataframe
        self.__testing_data = None #dataframe
        self.__predict_matrix = None #predictions cluster number dataframe
        self.__accuracy_matrix = None #accuracy dataframe
        self.__accuracy = None #prediction accuracy in float
        self.__training_frac = None #training fraction
        self.__testing_frac = None #testing fraction
        self.__pca = None #pca matrix
        return self
    
    def clear_cache(self):
        """clear the archive cache
        """
        self.__archive = pd.DataFrame()
        return self

    #endregion



    

    
