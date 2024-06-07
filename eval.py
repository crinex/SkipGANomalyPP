"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
python  eval.py                                    \
        --model skipganomaly           \
        --dataset folder                           \
        --resume ./lib/models/pth \
        --abnormal_class abnormal   \
        --batchsize 1             
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    model.test()

if __name__ == '__main__':
    main()