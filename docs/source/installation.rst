Installation
============

Installation and dependencies
-----------------------------
This project depends on several third party libraries, including:

**numpy**   (https://numpy.org/) (version>=1.22.2)

**nibabel** (https://nipy.org/nibabel/)

**nilearn** (https://nilearn.github.io/stable/index.html) (version>=0.9.0), ...

**nitools** (https://github.com/DiedrichsenLab/nitools)

.. code-block:: console

   $ pip install numpy nibabel nilearn neuroimagingtools

Or you can install the package manually from the original binary source as above links.

Additionally it uses Functional_Fusion tools:

**Functional_Fusion** (git clone https://github.com/DiedrichsenLab/Functional_Fusion.git)

Once you clone the functional fusion and selective recriuitment repositories, you may want to it to your PYTHONPATH, so you can import the functionality. Add these lines to your .bash_profile, .bash_rc .zsh_profile file...

.. code-block:: console

   $ export PYTHONPATH=<abspath_of_repo_maindir>:$PYTHONPATH