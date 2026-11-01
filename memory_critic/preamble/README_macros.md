## Organization of the Macros File 

to include all macro files, just use

``\input{preamble/_preamble_includes}''.

The "_preamble_includes" file contains all the relevant packages

files below ordered in listed of recommended inclusion:

- color_defs.tex = all the custom colors. Use this to set custom colors for links, refs, urls, and theorem headers

- header.tex = all the packages and basic formatting

- theorem_styles.tex = definition of all theorem styles and other environments

- characters.tex = character macros

- general_macros.tex = macros that should appear broadly across projects, irrespective of topic

- specific_macros.tex = Macros particular to a given project. left empty by default.

- algnames.tex = Formatting of algorithm names and other objects used in the paper

- coloredits.sty = A package for automating edit commands by other authors

- comment_macros.tex = a place to add further comment macros and author names for color edits

  