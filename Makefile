scripts := $(wildcard demo/*.py)
notebooks := $(patsubst %.py,%.ipynb,$(scripts))
notebooks-rendered := $(patsubst %.py,%.html,$(scripts))

.PHONY: all clean presentation notebooks rendered-notebooks

all: presentation notebooks

presentation: skrub-presentation.html

notebooks: $(notebooks)

notebooks-rendered: $(notebooks-rendered)

skrub-presentation.html: skrub-presentation.qmd
	quarto render $<

recipe-graph.svg: recipe-graph.dot
	dot -Tsvg $< -o $@

%.ipynb: %.py recipe-graph.svg
	jupytext $< --output $@

%.html: %.ipynb
	jupyter nbconvert $< --execute --to html

clean:
	rm -f demo/*.ipynb
	rm -f skrub-presentation.html
