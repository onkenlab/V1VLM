# V1VLM - An AI Scientist for analyzing neural coding in primary visual cortex

Code for the V1VLM model, a vision language model using
[ViV1T](https://github.com/bryanlimy/ViV1T-closed-loop) as a digital twin of
primary visual cortex, based on Google's Gemma-3 and Black Forest Labs'
FLUX.1.


## Installation

After setting up [ViV1T](https://github.com/bryanlimy/ViV1T-closed-loop)
install the following additional required packages:
```
pip install diffusers sentencepiece transformers accelerate fpdf
```


## Usage

Use a text context file to tell the model about relevant literature.
Then start a study for a given context file:
```
python run.py --context-file "./context.txt"
```
The model will come up with new neural coding hypotheses, generate input
images, run the digital twin to produce neuronal responses,
analyze the findings and conclude the study with a PDF report.
