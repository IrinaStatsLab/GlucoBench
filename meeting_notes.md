# Wednesday March 29, 2023
Next Meeting: Wednesday
Next Meeting Notetaker:

## Tasks
* Everyone: finish their sections in the paper. Prioritize writing over creating figures since then Renat/Dr. G can edit

* Renat
	* Upload TFT results 

* Lizzie:
	* Update the tables

* Nathaniel
	* Add a smooth trendline w/ confidence intervals on the "difficulty" plot, have T2, Healthy, etc. in the x label 
	* Use CI instead of boxplots


## Assigned Plots


# Wednesday March 22, 2023
Next Meeting: Wednesday March 29th, 2023    
Next Meeting Notetaker: **Nathaniel**

## Tasks
* Finish drafting your sections - 3/4 of a page or more
	* include a list of figures/tables for your section
* Create your assigned plots (see below)

## Assigned plots
* Nathaniel:
	* Plot all model errors without confidence intervals for each dataset ordered by size - i.e. y axis is error, x axis is datasets by size
	* One plot each for with and without covariates
* Nicky:
	* Plot best model error with confidence intervals for each dataset ordered by size - y axis is error, x axis is datasets by size
	* One plot each for with and without covariates
* Valeriya:
	* Plot all model errors without confidence intervals for each dataset ordered by type - y axis is error, x axis is datasets by type
		* should be Hall (none) -> Colas (mixed) -> IGLU (type 2) -> Dubosson (type 1) -> Weinstock (type 1), not sure on Dubosson/Weinstock ordering
	* One plot each for with and without covariates
* Lizzie:
	* Plot best model error with confidence intervals for each dataset ordered by type
		* See Valeriya's
	* One plot each for with and without covariates

## Updates
* Splitting updated to take test from the very end - prevent test data chronologically before train
* Subsampling had a bug in darts, Renat has fixed for us

# Wednesday March 8, 2023
Next Meeting: Wednesday March 22th, 2023  
Next Meeting Notetaker: **Nathaniel**

## Tasks
* **Update Overleaf paper with all model results**
* Valeriya:
	* For linear regression, change missing values to mean instead of zero (if successfully fixes OOD, then Nicky implement on Hall)
* In exploratory analysis, check distribution of covariate columns for each data split (e.g. train, test, validation)
	* run 'formatter.train_data' / 'formatter.test_data' / 'formatter.validation_data'
	* Plot the distribution

## Paper Assigned Sections
* Nathaniel: 5.2, 5.3
* Valeriya: 5.1, 3.2
* Nicky: 4.2, 3.3
* Lizzie: 3.1, 4.3

# Wednesday March 1, 2023
Next Meeting: Wednesday March 8th, 2023  
Next Meeting Notetaker: **Nicky?**

## Goals
* By **Saturday**:
	* Encode time column in column_definition (See Renat's yaml to be pushed)
	* Run linreg, xgboost, tft with covariates
	* Draft bullet points for your sections in overleaf (sections below)
* By **Wednesday**:
	* Run nhits, transformer with covariates
	* Write a paragraph for your sections in overleaf

## Paper Assigned Sections
* Nathaniel: 5.2, 5.3
* Valeriya: 5.1, 3.2, 
* Nicky: 4.2, 3.3 (if Renat qualifies what he means), 4.1 if he wants to try; or help with 5.*
* Lizzie: 3.1, 4.3

## Updates
* Lizzie added covariates, ran linreg and xgboost
	* Need to add encoded time
* Nathaniel has no covariates, ran linreg and xgboost
	* Need to add encoded time
* Valeriya has added covariates but lots of NAs
	* For example fast and slow insulin some subjects have a lot of NAs
	* Forward propagate calories until next known measurement 
		* Linearly interpolate to 0
	* Forward propagate insulin (fast lasts 2-3 hours, slow lasts 24 hours)
		* Linearly interpolate to 0
	* Balance and quality take sum for each subject
	* HR, BR, posture, activity, HRV interpolate between known values if possible
		* Set NAs as 0
	* Coretemp interpolate and fill endpoints with mean
	* Overall if subject is completely lacking, then code as 0
* How to report errors to report in paper:
	* Arima and linreg look at median
	* Other models look at Key: median{‘mean’}



# Wednesday February 22, 2023
Next Meeting: Wednesday March 1st, 2023
Next Meeting Notetaker: **TBD**

## Updates
* Nathaniel:
	* ran ARIMA, XGBoost, LinReg, TFT, NHits
	* Specific To Dos:
		1. Run transformer
		2. Add values to the Overleaf paper
		3. Add tim
	* Iglu has time which is a covariate: minute, second, etc.
* Valeriya:
	* Errors for ID > OOD surprisingly. Why?
		- The OOD looks like it's in the same timeframe as the training data.
		- There's only one subject in OOD. That subject's data may be very similar to another subject in the training data.
		- Maybe the test OOD has significantly smaller magnitudes so the relative error is much, much smaller. Would we do subject specific scaling?
			- previous studies have _more adhoc_ scaling: max to 450 and do min/max per subject. => absolute vs. relative error
	* Specific To Dos:
		1. train one model/subject. Test on ood 
		2. change the splits so have 2 subjects in testing data
		3. Try turning off scaling from darts
* Nikki:
	* Ran most models. Laptop kept turning off so getting Colab Pro+. Will finish running models by tomorrow morning
* Lizzie
	* TFT didn't run but everything else is in the Overleaf paper
	* Issue: OOD smaller than ID as well. However, b/c there's more subjects.

## TODOs
* Everyone
	* Saturday Feb 25th: update config files w covariates for the data set
		* data_type is either *categorical* or *
		* input_type is in either observed_input (time features - know in past and know what it will be in the future), known_input (heartbeat - know when we measure it), static_input (age, marital status, annual income - will likely not change)
	* Wednesday March 1st: finish up no-covariates models
	* Wednesday March 1st: run Linear regression + XGBoost w/ covariates

## Learnings
1. Set scaling to "None" in your config file b/c darts already does scaling.
2. In NHits, transformer, & TFT, set GPU devices to [0] if using Colab
3. What do the garbage errors mean?
4. The Google Colab Pro+ instance is still running even if the GUI doesn't show it

### New Models: Run NHits and Transformer Models
* These are time intensive - start early
* Reach out to Dr. Gaynanova or Renat for information about the models

### Continue Reading Paper to Summarize

### Some Sanity Checks for Model Error
- For any other model except arima, ID error should not be grater than OOD
- Glucose values range from 0-400, so error should be reasonably in range, otherwise model probably failed catastrophically

# Wednesday February 15, 2023
Next Meeting: Wednesday February 22nd, 2023
Next Meeting Notetaker: Nathaniel

## Updates
Arima model has new testing fw - no longer using Optuna
## TODOs
### Rerun Arima, XGBoost, and Linear Regression
* Upload models output by Saturday 2/18.
* Add the results to the Overleaf paper
* Some sanity checks: Ensure OOD is <= ID

### New Models: Run NHits and Transformer Models
* These are time intensive - start early
* Reach out to Dr. Gaynanova or Renat for information about the models

### Continue Reading Paper to Summarize
* Focus on the authors' _contributions_, usually found in the Abstract + Introduction
* Ask
	* Why was this paper chosen?
	* What is this sentence/unique word choice doing in terms of the context?
	* What are some shortcomings that are not apparent?
* Goal: understand what makes papers get published to use in writing our paper well

### Some Sanity Checks for Model Error
- For any other model except arima, ID error should not be grater than OOD
- Glucose values range from 0-400, so error should be reasonably in range, otherwise model probably failed catastrophically

# Wednesday February 8, 2023
Next Meeting: Wednesday February 15th, 2023
Next Meeting Notetaker: **TBD**

## Updates
**- Start using Colab**

## TODOs
### Add Covariates
**- Preprocess covariate data**
- Write processing steps
- Check if some features are missing (e.g. year, month, day, time, glucose, NAs, etc.)

### New Models
**- Linear Regression *(results should be a little worse than XGBoost)***

**- XGBoost (Gradient Boosted Trees)**
- Play around with hyperparameters to optimize XGBoost

### Paper
**- Choose a paper**
- look at particular points that make it interesting
- think like a reviewer to understand why the paper was chosen and stands out from other papers
- write ~5+ things highlightingn these points
- most important parts: Abstract -> 2 paragraphs of introduction -> results
- Questions to think about: "Why did the authors write this sentence?" "What is this sentence doing in the terms of the context?" "What is it doing?"
- Goal is to be able to transfer understanding to writing our paper well
**- Start adding results to overleaf paper, note about error computation/distribution**

**- Decide and text Renat what kind of section you want to work on in writing the paper (e.g. data, models, metrics, testing protocol, etc.)**

### Some Sanity Checks for Model Error
- For any other model except arima, ID error should not be grater than OOD
- Glucose values range from 0-400, so error should be reasonably in range, otherwise model probably failed catastrophically

# Wednesday February 1st, 2023
Next Meeting: Wednesday February 8th, 2023
Next Meeting Notetaker: **Nicky**

Slack updates due on **Monday before noon**

## Updates
- New (updated) interpolation function 
- Switching from Optuna to AutoARIMA 

## TODOs
- Modify .yaml files with the parameters set during the exploratory analysis
- Go to bin -> Create new script for your dataset (copy arima_weinstock.py but change the links) -> Go through the code and make sure you understand everything -> Run the script on your dataset, e.g. finish ARIMA model part
- Add the covariates to your dataset (if you have any)
- Think about Google Colab Pro+


# Wednesday January 25th, 2023
Next Meeting: Wednesday February 1st, 2023
Next Meeting Notetaker: **Valeriya**

Slack updates due on **Monday before noon**

Try to have independent team meeting sometime in the week (Nicky, Nathaniel, Lizzie, Valeriya - optional)

## Updates
- Welcome Nathaniel to the team!
- Switching to darts for models

## TODOs
- **NEW dataset assignments**:
	- Dubosson - Renat
	- Hall - Nicky
	- iglu - Nathaniel
	- Weinstock - Valeriya
	- Chase - Dr. Gaynanova
	- TBD - Lizzie
- Replicate dubosson.ipynb for your dataset
- Add covariates as columns to raw data csv
	- covariate data to be shared by Lizzie for your datasets


# Friday December 2nd, 2022
Next Meeting NEXT SEMESTER

## Updates
- Last meeting of the semester. Project will continue next semester, should express interest in continuing and then a decision will be made
- Will most likely switch to DARTS library for models (https://unit8co.github.io/darts/)
- End goal is to create a unified library of glucose datasets to establish a baseline for new models to be tested with

## TODOs
- Anonymous end of semester assessment


# Friday November 18, 2022
Next Meeting Friday December 2, Room 403

Next Meeting Notetaker: Nicky

## Updates
- Fixed bugs in LSTM

## TODOs
- Should code our own version of scalers called by the scale function in utils
- If scaling doesnt solve overfitting, look into regularlization through weight decay
- Look into scheduler to modify learning rate exponentially
- Implemenet early stopping using a counter while training.
- For scaler:
	- Change selection criterion for types we specify
	- Fit separate scaler for each column
	- Be able to pass in different scalers into the function in the parameters
	- Another parameter to scale curve wise or single mean/std/max/min
- Make a table in markdown w/ results of training w/ their configs
	- Keep track of the last epochs as well


# Friday November 11, 2022
Next Meeting Friday November 18, Room 403

Next Meeting Notetaker: Nicky

## Updates
- Urject wrote scaler and implemented
- Akhil experimented with different scalers
- Lizzie ran linear model and MLP 
- Nicky out sick

## TODOs
- Get MLP running in _nn notebook (Urjeet, Akhil, Nicky?)
- Edit scaling function to add options for scaling type (Urjeet)
- Run MLP and LSTM models that Renat pushed (All)
	- dropout should be between 0 and 1, controls percentage of data zero-ed out
	- document your findings for hyperparameter searching
		- track test error, table and/or graph
		- test loss may need to be weighted for batch size
	- MLP is baseline, LSTM is recurrent (accounts for time dependence)
	- Use LSTM_SeqtoSeq for LSTM model
		- may need to reshape data, message Renat if need help debugging issues
	- start using yaml to change configs
- Clean up and document your notebooks (All)
	- To make more user-friendly
		- Will be published eventually, also for own benefit later
	- Remove Renat's comments and write your own
- If thave time, start looking into Optuna (All)



# Friday October 14, 2022

Next Meeting: Friday October 21, Room 403

Next Meeting Notetaker: Urjeet

## Updates
- Dubosson and Weinstock finished implementing interpolate and split
- Hall and iglu needed debugging - done in meeting 

## TODOs
- Change min_drop_length to 144 (12 hours if 5 minute intervals) 
- If not done yet, round and average duplicates to get a perfect, unique grid in data_formatter 
- Encode subject as numeric - i.e. 0, 1, 2, ...
- Get all features from datetime timestamp - year, month, day, hour, minute
	- See code in slack from Nicky

# Friday October 7, 2022

Next Meeting: Friday October 14th, Room 403

Next Meeting Notetaker: ~~Urjeet~~Lizzie

## Update
- Interpolate function was completed with a few bugs which were later cleared up
- Renat combined Lizzie's and Urjeet's version of the interpolation fucntion to make an effecient one
- Split function logic was converted to an algorithm

## Notes
- The split function algorithm was designed and developed but had an error with the distribution ratio. It was fixed to some extent but the fix needs to be verrified
- Split functionality needs to be implemented at the right place, Renat will provide further details on this
- Set about 10% of the data as test and the remainder could be broken up into train and validation. Ex. if there were 8 subjects, 1 entire subject data would be test and 10% of each subject would be included in the test dataset. [validation set is generally equal to or greater than the size of test set]
- Important to remeber that we need to first split the raw data tables and then have TSDataset process the split datasets i.e. trian, test, & validation
- Interpolation function seems to have an error with datetime conversion in the dataset, Renat will follow up on this bug
- The goal is to have interpolation and split funcitons in place before next meeting to move ahead to scaling and wrap up pre-processing

## TODOs

- Make sure all the formatters are in place for the each of the dataset (All)
- Implement interpolation and split functions for each dataset (All)
- Make sure that the dataset example notebooks are update to date.Are structured as described by Renat and test the interpolate and split functions (All)

* * *
# Friday September 30, 2022

Next Meeting: Friday October 7th, Room 403

Next Meeting Notetaker: Akhil

# Updates
- Finished interpolate function in utils.py
- Added dataset formatters, yaml, etc for Dubosson and Hall

# Notes
- Renat to post a commit/push schedule today to be followed throughout the week
- No covariates added yet, later static covariates can be propagated, dynamic interpolation tbd
- Segment standardization/sampling is already implemented in ts_dataset


# TODOs
- Drop rows/subjects in \_\_init\_\_ of dataset formatter if applicable
	- Include a comment on which subjects/why dropped
- Write split_data function to split train, val, test
	- Define minimum segment length (tentatively min_length = 24 hours)
	- Split subject ids 90:10 train:test
	- Within each subject-segment in 90% train set, split as follows:
		- If segment length >= 3*min_length, split last min_length as test, next to last min_length as validate, rest as train
		- Else if segment length >= 2*min_length, split last min_length as validate, rest as train
		- Else assign as train only
- Check splitting proportions - i.e. what percent is train, val, test
	- Can adjust min_length based on this 

# Tentative task assignments:
- Urjeet push draft code for split_data
- Nicky will take over split_data for Monday/Tuesday
- All make sure your dataset is up to date with all functions we've added
	- In particular check that interpolate runs
	- Throughout the semester, you are responsible for supporting your dataset

* * *
# Friday September 23, 2022

Next Meeting: Friday September 30th, Room 403

Next Meeting Notetaker: Lizzie

## Updates
- Visualizations and DataFormatters for different datasets have been done.
- Work in interpolation has started.
- Various datasets have been preprocessed for use by DataFormatters.

## Meeting Nodes
- For Anderson, there are 3 time columns which is kind of weird.
- Disregard segments where there is just 1 datapoint (impossible to interpolate)
- Should have a seperate id column for segment
- Everything that processes the data should be in the same "tree"
- Temporarily ignore the issues with Dubusson, see if we can make it work


## Next Steps
- Finish up the interpolation part
- Interpolation function should be in the data formatter
- Call interpolation function at the beginning of the TSDataset
- Scalers and encoders

![image](https://user-images.githubusercontent.com/42304832/191999620-3147ee7f-e7ec-4968-887d-ed8d79358cc5.png)

* * *
# Friday September 16, 2022

## Notes
We will split data into segments by 45 minute gaps. Do linear interpolation on all other gaps (< 5) within segments. Then look at final length of segments.

Additional Info for *.yaml* file parameters,
- History/Input: num_encoder_steps (168 glucose values)
- Total length: 192 glucose values
- Predicted Steps: 24 glucose values

## TODOs

- Try working at data formatter with existing datasets (**All**)
- Explore new subject datasets + histograms, etc. (**All**)
- Work on segment splitting
- Adapt data formatter to subject datasets (**All**)
	- See ts_dataset.py and electricity example
- Start writing questions about research papers

* * *
# Friday September 9, 2022

Next Meeting: Friday September 16th, Room 403

Next Meeting Notetaker: Nicky

## Updates
In the last meeting we tlaked about looking at IGLOO data and using pytorch function on it to familiarize the team wiht the structure.
- The IGLOO dataset has been processed and added to the github
- Basic vizualization of the data has been completed
- We brainstormed couple of ways to rackle the segment gaps: in coclusion the best way was to draw histograms highlighting the distribution of aps between timestamps
- The table outlining all the features of a dataset has been created and is being updated with other datasets and their features
- Through discussion it was concluded that it would be great to have more description in the tables, like a footnote explaining all the abbrevations and a seperate   	table highlighting all the covariates
- Dynamiz covariates could use some more context as well. Example, if a study has controlled meals, we know what was eaten & when. Description of how these meals are 	defined, whether they are a vector of 3 numbers, and whether they are alligned to the time stamp woud aid the study
- Electircity example will be uploaded to github that can be used by the team to analyze how a loader class works. The primary focus for this week is on initializing 	the class, preprocessing data, and defining columns.

## ts_dataset Class
The class is designed to randomly sub sample a given dataset to give mini samples for a function to run stochastic gradient on. The high level idea is that it is a pipeline where you plug in a dataset and at the end of it you get samples in wrapper classes.


## TODOs

- Add classification information to the data table indiciating whether a dataset contains continuous or categorical data in it
- Possibly add a footnote for abbrevations such as BMI, OGTT etc
- Create a second table containig all covariates along with their description and refer to them from the first table through index reffering
- Dubasson Data set: Just include everything in it and describe it
- Create Histograms highlighting gaps in observation
	- Create histograms for all 5 subjects of the IGLOO dataset
- Write formatters, get_item,col_def functions in a class for IGLOO data following electricity class example (Split 3 data sets)
- Spend time studying Data Loader

* * *
# Friday September 2, 2022

Next Meeting: Friday September 9th, Room TBD

Next Meeting Notetaker: Akhil

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
