# Design descriptions and details for used Python package template

> Author: Henry Webel

[packaging.python.org](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
has an excellent tutorial on how to package a Python project. I read and used insights
from that website to help create the template which is available on GitHub at
[biosustain/python_package](https://github.com/biosustain/python_package), which was used
to build this project.

See the notes
[here](https://python-package-template-biosustain.readthedocs.io/developing.html).

## API principles

- main entry points should operate in the linear space (e.g. OD, cell density, etc.)
- secondary entry points can operate in the log space (e.g. ln(OD), ln(cell density),
  etc.)
- phenomological models should be used to fit the log space data
- mechanistic models are used to generate data or for fitting models in the linear space
  to curves without a lag phase

| parameter                   | mechanistic model | phenomological model |
| --------------------------- | ----------------- | -------------------- |
| lag phase                   | no                | yes                  |
| initial starting conditions | yes               | no (estimate: min(N))|
