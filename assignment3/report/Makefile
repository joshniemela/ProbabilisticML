# Latex Makefile using latexmk
# Modified by Dogukan Cagatay <dcagatay@gmail.com>
# Originally from : http://tex.stackexchange.com/a/40759
#
# Change only the variable below to the name of the main tex file.
DOCNAME=main

# Notebooks used to generate plots / converted to python code
WEEK5_GP_NOTEBOOK=../src/week5_gp.ipynb

# Plots generated from notebooks
PLOTS_W5_GP=./figures/gp_predict_gaussian.pdf ./figures/gp_predict_special.pdf

# Python code converted from notebooks
CODE_W5_GP=../src/week5_gp.py

# add any plots and code generated from notebooks here
PLOTS=$(PLOTS_W5_GP)
CODE=$(CODE_W5_GP)

# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: $(DOCNAME).pdf all clean

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: $(DOCNAME).pdf

# Convert jupyter notebooks to python files.
code: $(CODE)

# Run jupyter notebooks to create plots
plots: $(PLOTS)

$(PLOTS_W5_GP): $(WEEK5_GP_NOTEBOOK)
	jupyter-execute $<
	cp ../src/$(notdir $@) $@

$(CODE_W5_GP): $(WEEK5_GP_NOTEBOOK)
	jupyter-nbconvert $< --to=python
	sed -i '/^# In\[[0-9]*\]:/{N;N;d;}' $@

# MAIN LATEXMK RULE

$(DOCNAME).pdf: $(DOCNAME).tex
	latexmk -pdf -lualatex -interaction=nonstopmode -use-make -shell-escape $<

clean:
	latexmk -C

clean-code:
	rm -f $(CODE)

clean-plots:
	rm -f $(PLOTS)
