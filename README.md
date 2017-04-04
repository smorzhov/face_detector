# Face detector

Application that recognizes people faces using Eigen or Fisher algorithms

## Prerequisites

You will need the following things properly installed on your computer.

* [Python](https://www.python.org/)

## Installation

* `git clone https://github.com/smorzhov/tent_map.git` this repository
* `cd tent_map`
* `pip install -r requirements.txt`

## Running

Training:     `main.py -t [--train] (-e [--eigen] | -f [--fisher]) <user_name>`

Recognizing:  `main.py -r [--recognize] (-e [--eigen] | -f [--fisher])`

You may also specify camera id (by default it is 0) `-c [--camera-id] <id>`

For more information, please, see help message (`python main.py -h`)