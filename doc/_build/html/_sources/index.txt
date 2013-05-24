.. tract_querier documentation master file, created by
   sphinx-quickstart on Sun Mar 31 19:25:35 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WMQL's documentation!
=========================================

The White Matter Query Language (WMQL) is a technique to formally describe white matter tracts and to automatically extract them from diffusion MRI volumes. This query language allows us to construct a dictionary of anatomical definitions describing white matter tracts. The definitions include adjacent gray and white matter regions, and rules for spatial relations. This enables the encoding of anatomical knowledge of the human brain white matter as well as the automated coherent labeling of white matter anatomy across subjects.

This is an implementation of the WMQL language presented in



Click here for :ref:`example_script`





.. toctree::
   :maxdepth: 1

    tract_querier


WMQL Language Definition
------------------------

Backus–Naur Form of WMQL
::
        <WMQL Query> ::= <module import> | <assignment>
     <module import> ::= "import" <string>
        <assignment> ::= <identifier> "=" <expression>
        <expression> ::= <literal> 
                      |  <expression> <binary operator> <expression>
                      |  <unary_operator> <expression>
                      |  "("<expression")"
                      |  <function name>"("<expression")"
  <binary operator> ::= "or"
                      |  "and"
                      |  "not in"
    <unary operator> ::= "not"
     <function name> ::= "anterior_of"
                      |  "posterior_of"
                      |  "superior_of"
                      |  "inferior_of"
                      |  "lateral_of"
                      |  "medial_of"
                      |  "endpoints_in"
           <literal> ::= <identifier> | <number>
        <identifier> ::= <string>
                      |  <string>"."<hemisphere>
        <hemisphere> ::= "left"
                      |  "right"
                      |  "side"
                      |  "opposite"
            <string> ::= [a-zA-Z][a-zA-Z0-9_]*
           <number>  ::= [0-9]+


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

