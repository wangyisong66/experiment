from util import check_path, set_seed
from runmodel import RunModel
import logging


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SC_DSC')
    parser.add_argument('--db', default='brain', choices=['brain', 'ESC', 'mESC', 'mBladder', 'tc'])
    args = parser.parse_args()
    print(args)

    db = 'mBladder'  # = args.db
    check_path()
    logging.basicConfig(level=logging.DEBUG, filename='./results/' + db + '.log', filemode='a')

    set_seed(123)

    run = RunModel(db)
    # run.train_raw_ae()
    # run.train_raw_dsc()

    # run.train_imp_ae()
    run.train_imp_dsc()

    # for run.cfg.weight_coe2 in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
    #     print(run.cfg.weight_coe2)
    #     run.train_imp_dsc()
