# Friday September 2, 2022

Next Meeting: Friday September 9th, Room TBD

Next Meeting Notetaker: Nicky

## Intro and Logistics

Dr. Irina Gaynanova (Faculty Advisor): Associate Professor, Department of Statistics

Renat (Team Lead): 3rd year PhD student, Department of Statistics

Logistical notes:

- Communication via Slack - glunet channel, DMs   
- Meetings to be Friday 10a always, Room TBD

## Project Overview

Overall Goal: Machine learning for glucose prediction  
- Data from Continuous Glucose Monitors (CGMs)
- Main tool will be PyTorch

Metrics for performance: 
- Average percent error across prediction window
- Mean squared error across prediction window
- Possibly others tbd


## TODOs

- Export iglu dataset as CSV (**Lizzie**)
	- Will be uploaded to raw_data folder in GluNet
- Identify covariates from iglu dataset (**All**)
	- Determine/discuss segment cutoff (i.e. max gap length allowed), to be discussed in next week's meeting
- Learn/explore PyTorch (**All**)
	- Read papers in GluNet/papers
	- Investigate starter code - see GluNet/example_dataset
	- If needed, install VSCode and Copilot
- Load iglu data and wrap into dataset code (**All**)
	- See ts_dataset.py
- Add 3 datasets from Awesome-CGM to dataset table (**Lizzie**)

