#!/bin/bash

# Here is a little more information about the labels
# id: Unique identifier for each record or entry in the dataset
# person_age: The age of the person applying for the loan
# person_income: The annual income of the person applying for the loan
# person_home_ownership: The type of home ownership the person has (e.g., 'own', 'rent', 'mortgage')
# person_emp_length: The number of years the person has been employed
# loan_intent: The purpose or intent of the loan (e.g., 'education', 'medical', 'personal')
# loan_grade: A grade assigned to the loan based on the risk level (e.g., 'A', 'B', 'C')
# loan_amnt: The total amount of money being requested for the loan
# loan_int_rate: The interest rate applied to the loan, typically as an annual percentage rate (APR)
# loan_percent_income: The percentage of the person's income that is allocated to loan repayment
# cb_person_default_on_file: A binary indicator (yes/no) showing if the person has a history of defaulting on loans
# cb_person_cred_hist_length: The length of the person's credit history in years
# loan_status: The current status of the loan 0 or 1

# Here I will store all the comands for easier use and faster report afterwards

# First lets start with looking at the data and generating the histograms and scatter plots
#./display_data.py all --data-file data/train.csv
# Because everything is so polarized the scatter plot function doesn't work well for this data 
# and you can barely see at the top and bottom all the points. So for now I will just take a look
# at the histograms
