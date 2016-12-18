# Face detector
## Application that recognizes people's faces using Eigen or Fisher algorithms

Help:         `main.py -h`

Training:     `main.py -t [--train] (-e [--eigen] | -f [--fisher]) <user_name>`

Recognizing:  `main.py -r [--recognize] (-e [--eigen] | -f [--fisher])`

You may also specify camera id (by default it is 0) `-c [--camera-id] <id>`