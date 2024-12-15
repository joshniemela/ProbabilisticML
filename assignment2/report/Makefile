# Latex Makefile using latexmk
# Modified by Dogukan Cagatay <dcagatay@gmail.com>
# Originally from : http://tex.stackexchange.com/a/40759
#
# Change only the variable below to the name of the main tex file.
DOCNAME=main

WEEK3_PYRO_NOTEBOOK=../src/week3_pyro.ipynb
WEEK4_GP_NOTEBOOK=../src/week4_gp.ipynb

PLOTS_W3_PYRO=./figures/gmm_samples.pdf ./figures/cond_pdf.pdf
PLOTS_W4_GP=./figures/gp_samples.pdf ./figures/gp_predict.pdf
PLOTS=$(PLOTS_W3_PYRO) $(PLOTS_W4_GP)

CODE_W3_PYRO=../src/week3_pyro.py
CODE_W4_GP=../src/week4_gp.py
CODE=$(CODE_W3_PYRO) $(CODE_W4_GP)

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

./figures/gmm_samples.pdf: $(WEEK3_PYRO_NOTEBOOK)
		jupyter-execute $<
		cp ../src/$(notdir $@) $@

./figures/cond_pdf.pdf: $(WEEK3_PYRO_NOTEBOOK)
		jupyter-execute $<
		cp ../src/$(notdir $@) $@

./figures/gp_samples.pdf: $(WEEK4_GP_NOTEBOOK)
		jupyter-execute $<
		cp ../src/$(notdir $@) $@

./figures/gp_predict.pdf: $(WEEK4_GP_NOTEBOOK)
		jupyter-execute $<
		cp ../src/$(notdir $@) $@

$(CODE_W3_PYRO): $(WEEK3_PYRO_NOTEBOOK)
		jupyter-nbconvert $< --to=python
		sed -i '/^# In\[[0-9]*\]:/{N;N;d;}' $@

$(CODE_W4_GP): $(WEEK4_GP_NOTEBOOK)
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
