<h1 align="center">Customer Churn Prediction</h1>

<div align="center">

[![Language](https:/img.shields.io/badge/Python-darkblue.svg?style=flat&logo=python&logoColor=white)](https:/www.pyt/hon.org)
[![Framework](https:/img.shields.io/badge/sklearn-darkorange.svg?style=flat&logo=scikit-learn&logoColor=white)](https:/scikit-learn.org/stable/index.html)
![build](https:/img.shields.io/badge/build-passing-brightgreen.svg?style=flat)
![reposize](https:/img.shields.io/github/repo-size/Oyebamiji-Micheal/Prediction-of-Mohs-Hardness)
[![Topic](https:/img.shields.io/badge/End_to_End_ML-lightblue.svg?style=flat)]()
[![ML Tool](https:/img.shields.io/badge/MLflow-blue.svg?style=flat&logo=mlflow&logoColor=white)](https:/mlflow.org/)
[![ML Tool](https:/img.shields.io/badge/DVC-945DD6.svg?style=flat&logo=iterative&logoColor=white)](https:/dvc.org/)

</div>

<h4 align="center">An end to end machine learning project implementation with Azure deployment</h4>

<strong><p align="center">"Dinosaurs that failed to adapt went extinct. The same thing will happen to data scientists who think that training ML models inside Jupyter notebooks is enough." - Pau Labarta Bajo.</p></strong>

<br />

<img src="images/application-interface.jpg" width="580" height="679">

<h2>Table of Contents</h2>

- [Overview](#overview)
- [Objective](#objective)
- [Customer Churn and What it's all about](#customer_churn)
- [Dataset](#data)
- [MlFlow Integration](#mlflow)
- [Data Version Control (DVC)](#mlflow)
- [Azure](#azure)
- [Running Locally](#running_locally)

<a id="overview"></a>
<h2>Overview</h2>
<p align="justify">
This repository is an end-to-end machine learning project that focuses on predicting customer churn. It follows a comprehensive workflow that includes data ingestion, validation, transformation, model training, and model evaluation. The project aims to develop a predictive model that can identify customers who are likely to churn, allowing businesses to take proactive measures to retain them.
</p>

<a id="objective"></a>
<h2>Objective</h2>
<p align="justify">
Building on the foundational end to end workflow utilized in my previous project, <a href="https:/github.com/Oyebamiji-Micheal/Prediction-of-Mohs-Hardness/tree/main">"Prediction of Mohs Hardness"</a>, the objective of this project is to integrate the use of MLflow and DVC into my workflow. MLflow is a machine learning lifecycle management platform that enables tracking experiments, packaging code, and managing models. DVC (Data Version Control) is a version control system for machine learning projects that allows for efficient data and model versioning. By integrating MLflow and DVC, I aim to enhance my code reproducibility and efficient version control of my datasets and models.
</p>

<a id="customer_churn"></a>
<h2>Customer Churn and What it's all about</h2>
<p align="justify">
Customer churn refers to the phenomenon where customers stop doing business with a company or stop using its products or services. It is a critical metric for businesses, especially in industries with subscription-based models or recurring revenue streams.</p>

<img src="images/churn-image.jpg">

<p align="justify">Identifying customers who are likely to churn can help businesses take proactive measures to retain them, thereby reducing revenue loss and improving customer satisfaction.
</p>

<a id="data"></a>
<h2>Dataset</h2>
<p align="justify">
The dataset used for this project is obtained from <a href="https:/www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction">Kaggle</a>. It contains the following attributes:
</p>

- Customer ID: A unique identifier for each customer
- Surname: The customer's surname or last name
- Credit Score: A numerical value representing the customer's credit score
- Geography: The country where the customer resides (France, Spain, or Germany)
- Gender: The customer's gender (Male or Female)
- Age: The customer's age
- Tenure: The number of years the customer has been with the bank
- Balance: The customer's account balance
- NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
- HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
- IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
- EstimatedSalary: The estimated salary of the customer
- Exited: Whether the customer has churned (1 = yes, 0 = no)

<a id="mlflow"></a>
<h2>MlFlow Integration</h2>
<p align="justify">
To integrate MLflow into the project, I used Dagshub as my remote server where I can easily log and compare different experiments and also track the performance of my model. 

<img src="images/mlflow.jpg">
</p>

<a id="mlflow"></a>
<h2>MlFlow Integration</h2>
<p align="justify">
To integrate MLflow into the project, I have defined a YAML file that specifies the different stages of the pipeline. Each stage has a command (<code>cmd</code>) that runs a Python script, dependencies (<code>deps</code>) that are required for the script to execute, and outputs (<code>outs</code>) that are generated by the script. Additionally, some stages have parameters (<code>params</code>) and metrics (<code>metrics</code>) that are used for model training and evaluation, respectively.</p>

Here is an overview of the stages defined in the YAML file:

- <p align="justify"><strong>data_ingestion</strong>: This stage runs the <code>stage_01_data_ingestion.py</code> script, which is responsible for ingesting the data. The dependencies include the script itself, the <code>data_ingestion.py</code> component, the <code>config.yaml</code> file, and the output CSV file <code>Churn_Modelling.csv</code>.</p>

- <p align="justify"><strong>data_validation</strong>: This stage runs the <code>stage_02_data_validation.py</code> script, which validates the ingested data. The dependencies include the script, the <code>data_validation.py</code> component, the output CSV file from the previous stage, the <code>config.yaml</code> file, and the <code>schema.yaml</code> file. The output is a <code>status.txt</code> file indicating the status of the validation.</p>

- <p align="justify"><strong>data_transformation</strong>: This stage runs the <code>stage_03_data_transformation.py</code> script, which transforms the validated data. The dependencies include the script, the <code>data_transformation.py</code> component, the <code>status.txt</code> file from the previous stage, and the <code>config.yaml</code> file. The outputs include a preprocessor joblib file, and train and test CSV files.</p>

- <p align="justify"><strong>model_training</strong>: This stage runs the <code>stage_04_model_trainer.py</code> script, which trains a machine learning model. The dependencies include the script, the <code>model_trainer.py</code> component, the train CSV file from the previous stage, and the <code>config.yaml</code> file. The parameters for the model training are specified in the YAML file. The output is a trained model joblib file.</p>

- <p align="justify"><strong>model_evaluation</strong>: This stage runs the <code>stage_05_model_evaluation.py</code> script, which evaluates the trained model. The dependencies include the script, the <code>model_evaluation.py</code> component, the test CSV file from the previous stage, the trained model joblib file, and the <code>config.yaml</code> file. The metrics generated during the evaluation are stored in a <code>metrics.json</code> file.</p>

<a id="azure"></a>
<h2>Azure</h2>
<p align="justify">
The last step is to actually deploy this project. However, I could not deploy it because I am currently out of my student Azure subscription. If you have an Azure subscription, you can follow the steps below to deploy the project:</p>

1. Create an Azure Machine Learning workspace.
2. Set up the necessary resources such as compute instances, storage accounts, and container registries.
3. Build a Docker image of the project.
4. Deploy the Docker image to Azure Container Instances or Azure Kubernetes Service.


<a id="running_locally"></a>
<h2>Running Locally</h2>
<p align="justify">

### STEP 00 - Clone the repository

```bash
git clone https://github.com/Oyebamiji-Micheal/End-to-End-Customer-Churn-Prediction-using-MLflow-and-DVC
```

### STEP 01 - Create a virtual environment 

**Windows** (cmd) <br>

```bash
cd End-to-End-Customer-Churn-Prediction-using-MLflow-and-DVC
pip install virtualenv
python -m virtualenv venv
```

or

```bash
python3 -m venv venv
```

**macOS/Linux** <br>

```bash
cd End-to-End-Customer-Churn-Prediction-using-MLflow-and-DVC
pip install virtualenv
python -m virtualenv venv
```

### STEP 02 - Activate environment <br>

**Windows** (cmd)

```bash
venv\scripts\activate
```

**macOS/Linux**

```bash
. venv/bin/activate
```

or

```bash
source venv/bin/activate
```

### STEP 03 - Install the Requirements

Windows/macOS/Linux <br>

```bash
pip install -r requirements.txt
```



### STEP 04 - Run app.py

```bash
python app.py
```

Now,

```bash
Open the url: http://127.0.0.1:5000/ 
```

<br />

<img src="images/prediction-result.jpg">