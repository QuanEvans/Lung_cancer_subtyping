# Lung_cancer_subtyping



## Description of the project
The goal of this class project is to build and evaluate a mathematical model that can discriminate between two lung cancer subtypes.
To build the model we use an unsupervised k-means clustering algorithm (Euclidean distance) of 58 NSCLC tumors using k=2.
To evaluate the model we compute the model accuracy. Accuracy in this case is the percentage of samples that the model assigns to the wrong subtype outof all the samples it classifies. </br>
<img src = https://wikimedia.org/api/rest_v1/media/math/render/svg/7588abeafe63ab4b8ae63f954978186276e54d01 width = "400"/>



## Data
The data contains 40 adenocarcinoma (AD) samples and 18 squamous cell carcinoma (SCC) samples.

The data is available in the SOFT formatted family file available under the Download header at the following link.
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10245 </br>
The SOFT formatted gz file also avaible in the data folder of this repository.

## Packages
All packaes used in this project are in the Python language.
The packages used in this project are:
- pandas
- numpy
- GEOparse 
- skelearn

The GEOparse package is used to parse the SOFT formatted file and extract the data.
The sklearn package is used to perform the k-means clustering algorithm.
the following code is used to install the GEOparse package:
```cmd
$pip install GEOparse
$pip install sklearn
$pip install pandas
$pip install numpy
```

