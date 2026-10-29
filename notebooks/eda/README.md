# TMDB Exploratory Analysis

This project aims to perform exploratory data analysis on the TMDB movie dataset. The analysis includes data cleaning, statistical analysis, and visualizations to derive insights from the data.

## Project Structure

```
tmdb-exploratory-analysis
├── data
│   ├── raw                # Original, unprocessed data files
│   └── processed          # Cleaned and processed data files
├── notebooks
│   └── exploratory_analysis.ipynb  # Jupyter notebook for exploratory analysis
├── src
│   ├── data_processing.py  # Functions for loading and cleaning data
│   ├── analysis.py         # Functions for statistical analysis
│   ├── visualization.py     # Functions for creating visualizations
│   └── utils.py            # Utility functions used across modules
├── requirements.txt        # List of required Python packages
├── .gitignore              # Files and directories to ignore by Git
└── README.md               # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd tmdb-exploratory-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Place your raw data files in the `data/raw` directory.
- Run the data processing functions from `src/data_processing.py` to clean and prepare the data.
- Use the functions in `src/analysis.py` to perform statistical analyses.
- Generate visualizations using the functions in `src/visualization.py`.
- Refer to the `notebooks/exploratory_analysis.ipynb` for examples of how to use the functions defined in the source files.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.