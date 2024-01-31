# AP2-Social-media-data-for-better-local-forecasts

## Description

This project utilizes social media data to enhance local weather forecasts.
This project utilizes the [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b) model to process ER5 tweet datasets and includes a variety of scripts for data handling and testing. The repository contains a batch folder for efficient, parallel processing of large data sets, a bootcamp folder with unmodified reference materials, a script folder with reusable Python scripts for tasks like data formatting and plotting, and a test folder with scripts and notebooks for code verification and experimentation. This project represents a comprehensive effort to leverage social media data for meteorological insights.

### Installation

```bash
# Example installation steps
git clone https://github.com/rajhaq/AP2-Social-media-data-for-better-local-forecasts.git
cd AP2-Social-media-data-for-better-local-forecasts
```

### /batch

The `batch` folder contains scripts for running the Falcon 40B model to generate a dataset from the ER5 tweet dataset. This setup is designed to handle large volumes of data efficiently.

#### Structure

- The folder includes four sub-folders (folder 1, 2, 3, 4) each intended for running separate jobs in parallel to maximize throughput.

#### Description

- Each script in these folders is configured to process a specific subset of the data (e.g., 5000 data points).
- These scripts facilitate the batch processing of data using the Falcon model.

#### Usage

- Initially, use the test scripts to ensure basic functionality of the Falcon model.
- For final testing and execution, modify the `start_index` and `end_index` parameters to define the data range.
- Adjust the job run time (`SBATCH --time`) as needed based on data size and estimated processing requirements.

#### Example Commands

- Basic test command: `srun apptainer run --nv [container path] python3 test_basic_falcon_model.py`
- Command for processing data: `srun apptainer run --nv [container path] python3 falcon_script_for_5000_data.py`

### /bootcamp

The `bootcamp` folder contains files and resources used during a bootcamp. This folder serves as a reference and includes various materials related to the project.

- The folder holds data and scripts utilized in the bootcamp.
- No modifications have been made to the files in this folder; they are kept as-is for reference purposes.

### /script

The `script` folder contains a collection of reusable Python scripts designed to assist in various tasks such as data formatting, conversion, and plotting.

- This folder includes scripts for commonly used operations in data processing and analysis.
- Scripts cover functions like CSV data formatting, converting data formats, and generating plots.

#### Usage

- These scripts are designed to be modular and reusable, making them useful for handling routine data processing tasks.
- Each file contains their own description and example useable code

### /test

The `test` folder includes scripts and Jupyter Notebook files used for testing code during development.

- Contains various test scripts and notebooks that were used to verify and validate code before finalization.
- Although not critical for the project's main functionality, these files can be helpful for understanding the testing process and experimenting with code.

#### Note

- While the test files are not essential for the project's operation, they offer valuable context and examples for developers and users interested in the project's testing practices.
