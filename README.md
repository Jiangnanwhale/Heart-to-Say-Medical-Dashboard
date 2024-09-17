 Heart to Say 
==============================================
Group A — Project Management and Tools for Health Informatics — HT2024

## :heart: Group Members
`Ifani Pinto Nada`, `Mahmoud Elachi`, `Nan Jiang`, `Sahid Hasan Rahimm`, `Zhao Chen`  
## :innocent: Project Despribtion
The Heart Failure Mortality Prediction project aims to build a web-based medical dashboard that supports physicians to predict the risk of mortality due to heart failure. Physicians will be able to reassess treatment plans, explore alternative therapies, and closely monitor patients to help mitigate the risk of mortality. Prescriptive analytics will be used on patient data to help physicians identify specific factors contributing to elevated mortality risk. Thus, it will provide recommendations based on existing medical guidelines to guide in clinical decision-making on an individual basis for prevention and/or mitigation of mortality due to heart failure.

## :key: About the data 
| Attribute                | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| age                      | Age of the patient                                             |
| anaemia                  | Haemoglobin level of patient (Boolean)                         |
| creatinine_phosphokinase | Level of the CPK enzyme in the blood (mcg/L)                   |
| diabetes                 | If the patient has diabetes (Boolean)                          |
| ejection_fraction        | Percentage of blood leaving the heart at each contraction      |
| high_blood_pressure      | If the patient has hypertension (Boolean)                      |
| platelets                | Platelet count of blood (kiloplatelets/mL)                     |
| serum_creatinine         | Level of serum creatinine in the blood (mg/dL)                 |
| serum_sodium             | Level of serum sodium in the blood (mEq/L)                     |
| sex                      | Sex of the patient                                             |
| smoking                  | If the patient smokes or not (Boolean)                         |
| time                     | Follow-up period (days)                                        |
| DEATH_EVENT              | If the patient deceased during the follow-up period (Boolean)  |

Attributes having Boolean values: 0 = Negative (No); 1 = Positive (Yes)

Data source - https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data

## :framed_picture: Demo



## :dizzy: How to use?
<ol>
    <li>Clone this repo</li>
    <li>Install all the dependencies</li>
    <li>Download deepsort <a href="https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6">checkpoint</a> file and paste it in deep_sort_pytorch/deep_sort/deep/checkpoint</li>
    <li>Run -> streamlit run app.py</li>
</ol>

![Logo](assets/heart_to_say.png)
