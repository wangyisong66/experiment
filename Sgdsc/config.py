from configs.cfg_brain import ConfigBrain
from configs.cfg_ESC import ConfigEsc
from configs.cfg_mBladder import ConfigMBladder
from configs.cfg_tc import ConfigTC

config_list = {"brain": ConfigBrain,
               "ESC": ConfigEsc,
               # "mESC": ConfigMesc,
               "mBladder": ConfigMBladder,
               "tc": ConfigTC}


def get_config(name):
    assert name in config_list.keys()

    print("get configs", name)
    return config_list[name]


if __name__ == "__main__":
    cfg = get_config("orl")
    print(cfg)
