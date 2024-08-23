# Fine-Tuning on Pre-Training Data: A Model Comparison

The code for this project is entirely contained with the `main.ipynb` notebook. A notebook was chosen (as opposed to a `.py` file) because this project was run on Google Colab to take advantage of cloud GPU (which was very necessary for a fine-tuning project). The notebook annotates the code at length, so in the interest of giving the grader less to read, please see the notebook for explanations of the code.

## Installation

The notebook must be connected to a kernel; from there, the notebook itself contains a `pip install` command that installs all necessary packages (this was the easiest way to install packages on Google Colab). However, if you'd like to create an environment and install the packages into it, run the following commands in the directory that contains this file.

```
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

From there, you can select the virtual environment as your notebook's kernel and forgo the `pip install` command in the notebook.

Note that running this code on your own is highly contingent upon the ready availability of a robust GPU.

Also note that the two data sources for the notebook, `bible.txt` and `bible_qa_pairs.csv`, are included in this directory and require no installation or otherwise extra steps.
