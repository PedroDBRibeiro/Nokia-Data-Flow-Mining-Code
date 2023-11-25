import configparser


def load_config(env):
    config = configparser.RawConfigParser()
    config.read(env+'.properties')
    print(config)
    return config

def load_property(sect,prop, env= 'PYCELONIS\local'):
    c = load_config(env)
    return c.get(sect, prop)


