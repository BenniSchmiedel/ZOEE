.. image:: /paper/figures/Logo_long.png
    :scale: 50%

ZOEE is an python implementation of Zero to One dimensional Earth Energy balance models.

It serves as a framework to compile and simulate energy balance models from a collection of parameterizations,
which describe the energy transport of reduced/simplified earth system.

Applications
------------
The range of applications and features is constantly extended. They focus on:

* Exploration and quantification of parameterizations describing the climate system.
* Exploration and quantification of radiative climate forcings
* Parameter optimization tools

Documentation
-------------

A detailed documentation for ``ZOEE`` is hosted on `Read-the-docs <https://lowebms.readthedocs.io/en/latest/>`_.

Installation
------------

|PyPi|

The simplest way to install ``ZOEE`` is from PyPi.
To download and install ``ZOEE`` via ``pip`` you can use::

    pip install ZOEE
Alternatively, you can clone the git repository of the source code with::

    git clone https://github.com/BenniSchmiedel/ZOEE.git

and manually run the setup.py which installs the package with all its dependencies::

    python setup.py install

The most implementations are based on the work of former developers of climate models which are tried to be gathered and combined within this package.
The central approaches to formulate energy balance models included in this package are based on the publications from Sellers (1969) and Budyko (1968).

Contribution
------------

You are very welcome to work with this package and extend it to allow an application to anything you are interested.
If you are interested in contributing to this project or have problems with the usage, feel free to contact me:

Benjamin Schmiedel (mail: benny.schmiedel@gmail.com, github-username: BenniSchmiedel)
