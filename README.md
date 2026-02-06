# Replication package (FRL submission)

This repository contains code to reproduce the monthly panel construction, estimation, and the main tables/figures.

## Requirements
- Python 3.x
- Internet connection (needed to download FRED series)

## Files in this repository
- paper_main_2024.py : main pipeline script
- fed_data.csv : FOMC dates used to construct the tightening-news proxy
- requirements.txt : Python dependencies

## External data (not redistributed here)
This project uses publicly available data. The Shiller data file is not included in this repository.
Please download Robert Shiller Online Data (Yale University) and save the file in the repository folder
(same directory as `paper_main_2024.py`) using one of the following filenames:

- ie_data.xls - Data.csv
- ie_data.xls
- ie_data.csv

The script searches for the Shiller file using the filenames above.

## Install dependencies
pip install -r requirements.txt

## Run
python paper_main_2024.py

## Outputs
The script creates an output folder and saves the final analysis panel, tables, and figures to:
- final_results_2024_main/
