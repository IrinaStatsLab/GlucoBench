| Number | Covariate | Abbreviation/Description | Data type | Aligned with time scale | 
| --- | --- | --- | --- | --- | 
| 1 | age | - | continuous | - | 
| 2 | gender | - | categorical | - |
| 3 | BMI | body mass index | continuous | - |
| 4 | height | - | continuous | - |
| 5 | weight | - | continuous | - |
| 6 | race | - | categorical | - |
| 7 | A1c| Hemoglobin A1c (measure of long term blood glucose) | continuous | - |
| 8 | FBG | fasting blood glucose | continuous | - |
| 9 | OGTT | oral glucose tolerance test | continuous | - |
| 10 | SSPG | steady state plasma glucose (quantifies insulin resistance) | continuous | - |
| 11 | total cholesterol | - | continuous | - |
| 12 | HDL | high density lipoprotein ("good" cholesterol) | continuous | - |
| 13 | LDL | low density lipoprotein ("bad" cholesterol) | continuous | - |
| 14 | triglycerides | lipids (fats) found in blood | continuous | - |
| 15 | medications | varies | categorical | - |
| 16 | time | - | continuous | yes |
| 17a | Anderson meals | tbtime stamped vector of 2 numbers: meal size (g) and self-monitored blood glucose (SMBG) | continuous | no, but can be matched to nearest time |
| 17b | Hall meals | time stamped vector of 5 numbers: calories, fat, carbohydrates, sugar, fiber, protein all in grams | continuous | yes, but may need to check data processing |
| 18a | Anderson insulin | time stamped insulin amount delivered (see README on MonitorTotalBolus) | continuous | no, but can match to nearest |
| 18b | Dubosson insulin | time stamped vector of 2 numbers: amount of slow insulin and amount of fast insulin, units unknown | continuous | no, but can be matched to nearest time |
| 19 | Anderson ketones | time stamped ketone measurement (mmol/L) | continuous | no, but can match to nearest |
| 20 | Dubosson activity | time stamped activity (unknown units) | continuous | no, aggregated by the second |
| 21 | Dubosson BR | breathing rate: time stamped breathing rate | continuous | no, aggregated by the second |
| 22 | Dubosson HR | heart rate: time stamped heart rate | continuous | no, aggregated by the second |
| 23 | Dubosson ECG | electrocardiogram: waveform measurements (unknown units) | continuous | no, collected every ~0.004 seconds |

