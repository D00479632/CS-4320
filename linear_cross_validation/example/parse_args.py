# This is just an example of argument curtis added in the pipeline
parser.add_argument('--model-type',    '-M', default="SGD", type=str,   choices=["SGD", "linear", "SVG", "boost", "forest", "tree"], help="Model type")

