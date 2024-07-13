import sys
import nfeloqb

if sys.argv[1] == 'run':
    nfeloqb.run()

if sys.argv[1] == 'run_now':
    nfeloqb.run(force_run=True)