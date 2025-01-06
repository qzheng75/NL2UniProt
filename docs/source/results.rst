Results
=====

.. _benchmark:

Benchmark
------------

We evaluate the performance of our approach by perform inference on three levels of descriptions of sequences.
For each sequence, we can describe its properties with different levels of granularity. Here is an example:

Sequence with id ``A0A1B0GTW7`` (`UniProt Entry <https://www.uniprot.org/uniprotkb/A0A1B0GTW7/entry>`_) can be described:

- With high level of granularity:

  I want a protein with a specific function in the establishment of left-right asymmetry and involvement 
  in a complex disorder characterized by disruption of left-right organ placement, leading to randomization 
  of visceral organs and associated congenital defects. This protein, which is a type of metallopeptidase 
  with metal ion binding capabilities, also plays a role in cell adhesion and proteolysis.
                                 
- With medium level of granularity:

  I want a protein with metal-binding properties and the ability to break down proteins within cells, 
  playing a crucial role in the development of certain organs and potentially causing a rare genetic 
  disorder affecting the placement of internal organs.

- With low level of granularity:

  I want a protein with metal ion binding and peptidase activity properties.

With the current deployed version, we test on 2000 descriptions with the above three level of granularity.
The performances are listed in the table below.
                                 
