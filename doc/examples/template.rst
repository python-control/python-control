:orphan: remove this line and the next bbefore use (supresses toctree warning)

.. currentmodule:: control

**************
Sample Chapter
**************

This is an example of a top-level documentation file, which serves a
chapter in the User Guide or Reference Manual in the Sphinx
documentation.  It is not that likely we will create a lot more files
of this sort, so it is probably the internal structure of the file
that is most useful.

The file in which a chapter is contained will usual start by declaring
`currentmodule` to be `control`, which will allow text enclosed in
backticks to be searched for class and function names and appropriate
links inserted.  The next element of the file is the chapter name,
with asterisks above and below.  Chapters should have a capitalized
title and an introductory paragraph.  If you need to add a reference
to a chapter, insert a sphinx reference (`.. _ch-sample:`) above
the chapter title.

.. _sec-sample:

Sample Section
==============

A chapter is made of up of multiple sections.  Sections use equal
signs below the section title.  Following FBS2e, the section title
should be capitalized.  If you need to insert a reference to the
section, put that above the section title (`.. _sec-sample:`), as
shown here.


Sample subsection
-----------------

Subsections use dashes below the subsection title.  The first word of
the title should be capitalized, but the rest of the subsection title
is lower case (unless it has a proper noun).  I usually leave two
blank lines before the start up a subection and one blank line after
the section markers.


Mathematics
-----------

Mathematics can be uncluded using the `math` directive.  This can be
done inline using `:math:short formula` (e.g. :math:`a = b`) or as a
displayed equation, using the `.. math::` directive::

.. math::

     a(t) = \int_0^t b(\tau) d\tau


Function summaries
------------------

Use the `autosummary` directive to include a table with a list of
function sinatures and summary descriptions::

.. autosummary::

   input_output_response
   describing_function
   some_other_function


Module summaries
----------------

If you have a docstring at the top of a module that you want to pull
into the documentation, you can do that with the `automodule`
directive:

.. automodule:: control.optimal
   :noindex:
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: control

The `:noindex:` option gets rid of warnings about a module being
indexed twice.  The next three options are used to just bring in the
summary and extended summary in the module docstring, without
including all of the documentation of the classes and functions in the
module.

Note that we `automodule` will set the current module to the one for
which you just generated documentation, so the `currentmodule` should
be reset to control afterwards (otherwise references to functions in
the `control` namespace won't be recognized.
