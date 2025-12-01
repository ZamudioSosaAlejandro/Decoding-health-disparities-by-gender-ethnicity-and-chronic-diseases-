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

Additional packages are listed and version-controlled in the `renv.lock` file.

System requirements

Code was developed and executed on a Windows-based workstation with the following specifications:

Operating system: Windows  
CPU: Multi-core processor  
Memory: 80 GB RAM  

---

Installation instructions

All required R packages can be installed automatically using the `renv` package to ensure full reproducibility of the computational environment.

To install and restore the environment, run:

```r
install.packages("renv")
renv::restore()
```

Typical installation time ranges from 20 to 40 minutes, depending on existing packages and system configuration.

Data

Due to privacy restrictions and data use agreements, raw microdata from the national health surveys cannot be publicly shared in this repository. The data are available upon reasonable request to the corresponding author, as described in the Data Availability section of the manuscript.

Scripts for data harmonization, variable construction, and weighting using survey expansion factors are included in this repository. Intermediate and processed datasets used for figure and table generation are either provided or can be reconstructed using the included scripts.

Source data for all figures and tables presented in the manuscript are provided in the Supplementary Information/Data Source files accompanying the publication.

Code execution and run time

The analytical workflow consists of:

Descriptive weighted analyses of social determinants

Training and evaluation of random forest models by gender and ethnicity

Model-agnostic variable importance analysis based on missing AUC

Accumulated local dependence (ALD) plots

Sensitivity analyses stratified by country

Most scripts execute within several minutes. Model training and resampling for variable neutralization and sensitivity analyses may take up to one to two hours per subgroup, depending on system specifications.

A master script (e.g., Resultados_generales.R or run_all.R) is provided to execute the full analysis pipeline in sequential order.

Code description and help

The codebase is organized into modular R scripts corresponding to each analytical step, including data cleaning, model training, evaluation, and visualization. Inline comments and function documentation are provided to clarify the purpose and output of each script.

Model training relies on repeated cross-validation (10 repetitions, 3-fold cross-validation) using the caret framework, with weighted training and validation based on survey expansion factors. Random forest models are implemented using the ranger package, with hyperparameter tuning over mtry values ranging from 3 to 10.

Model performance is evaluated using the area under the receiver operating characteristic curve (AUC). Variable importance is assessed using a model-agnostic approach by measuring the average decrease in AUC after variable neutralization across 50 resamples.# Decoding-health-disparities-by-gender-ethnicity-and-chronic-diseases-
