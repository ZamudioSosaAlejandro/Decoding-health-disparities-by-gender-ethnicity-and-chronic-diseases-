
This repository contains the R code used in the analysis for the article: 

Decoding health disparities by gender, ethnicity and chronic diseases across three Latin American countries

Chronic diseases disproportionately affect specific ethnic and gender groups, yet the social determinants underlying these disparities in Latin America remain insufficiently understood. In this study, we analyze nationally representative health survey data from Brazil, Mexico, and Ecuador (2018–2019), covering a total weighted adult population of 96,726,891 individuals. Using random forest machine learning models, we predict chronic disease diagnoses based on education, occupation, and access to essential services such as sanitation, drinking water, and garbage collection.

Our results show that model performance is higher for indigenous and Afro-descendant populations, highlighting deep-rooted inequalities. For women, occupation and education emerge as the strongest predictors, reducing model performance by 8.57% and 7.36%, respectively, when neutralized. In contrast, occupation is the most critical predictor for men, decreasing model performance by 19.6% when controlled. These findings underscore the need for public policies that are specifically tailored to the social and structural conditions of different ethnic and gender groups.

The code in this repository allows both the reproduction of the analyses and figures presented in the manuscript and the application of the methodological framework to other population health datasets.

---

Installation and System Requirements

R software and packages used for analysis

All data processing, statistical analysis, and machine learning modeling were conducted using the R programming language.

R version: 4.4.0

Key R packages used include:

caret  
ranger  
DALEX  
tidyverse  
data.table  
survey  
pROC  
ggplot2  

System requirements

Code was developed and executed on a Windows-based workstation with the following specifications:

Operating system: Windows  
CPU: Multi-core processor  
Memory: 80 GB RAM  

Typical installation time ranges from 20 to 40 minutes, depending on existing packages and system configuration.

Data

This repository contains preprocessed data for analysis. The raw data and its sources are described in the Data Availability section.

This repository includes scripts for data analysis using machine learning models. The intermediate and processed datasets used to generate figures and tables are provided or can be reconstructed using the included scripts.

The source data for all figures and tables presented in the manuscript are included in the Supplementary Information/Data Source files accompanying the publication.

Code Execution and Runtime

The analytical workflow consists of:

Training and evaluation of random forest models by gender and ethnicity

Model-independent variable significance analysis based on missing AUC

Cumulative local dependency (ALD) plots

Sensitivity analysis stratified by country

Most scripts run in minutes. Model training and resampling for variable neutralization and sensitivity analyses can take one to two hours per subgroup, depending on system specifications.

Code Description and Help

The codebase is organized into modular R scripts that correspond to each analytical step, including model training, evaluation, and visualization. Inline comments and feature documentation are included to clarify the purpose and output of each script.

Model training is based on repeated cross-validation (10 iterations, triple cross-validation) using the caret framework, with weighted training and validation based on survey expansion factors, for subsequent application using ranger to increase code efficiency.

Model performance is assessed using the area under the receiver operating characteristic (AUC) curve. Variable significance is assessed using a model-independent approach, measuring the average decrease in AUC after variable neutralization in 50 new samples.