# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    	?= -W
SPHINXBUILD   	?= sphinx-build
SOURCEDIR     	= source/$(REQUESTED_DOCS_PKG)
BUILDDIR      	= build
DOXYGENDIR    	= doxygen_build
HTMLDIR       	= $(BUILDDIR)/html
TTIR_BUILDER_BUILDDIR	= $(BUILDDIR)/ttir-builder
PORT          	?= 8888
DOCS_VERSION       	?= latest

# register_python_operation doesn't load the doc string when fast_runtime_mode is enabled.
# disable fast_runtime_mode to enable loading the doc string.
#TTNN_FLAGS      = TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
#GITHUB_TOKEN    ?= INSERT_TOKEN_HERE
#TTNN_API_DIR    = source/ttnn/ttnn/api

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help default clean html publish build_doxygen sphinx_build_dir html/ttir-builder

default: html

all: clean html server

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf $(DOXYGENDIR)
#	@rm -rf $(TTNN_API_DIR)

build_doxygen: clean
	@cd .. && doxygen docs/doxygen.cfg.in

sphinx_build_dir: clean
	@mkdir -p $(BUILDDIR)

html/ttir-builder: build_doxygen sphinx_build_dir
	@DOCS_VERSION=$(DOCS_VERSION) REQUESTED_DOCS_PKG=ttir-builder $(SPHINXBUILD) "$(SOURCEDIR)/ttir-builder" "$(TTIR_BUILDER_BUILDDIR)" $(SPHINXOPTS) $(O)

html: html/ttir-builder
	mkdir -p $(HTMLDIR)
	mv -f $(TTIR_BUILDER_BUILDDIR) $(HTMLDIR)/ttir-builder
	cp source/index.html $(HTMLDIR)/


server:
	@echo "Navigate to: \033[4;33mlocalhost:$(PORT)/index.html\033[0m"
	@cd $(HTMLDIR) && python -m http.server $(PORT)
