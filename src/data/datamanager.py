import pickle
import os

pickle_path = 'resources/pickle/'

def save(obj, name):
    print(f"Saving {name}")
    with open(f'{pickle_path}{name}.pkl', 'wb') as out:
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)
        print(f"> {name} saved")


def load(name):
    print(f"Loading {name}")
    with open(f'{pickle_path}{name}.pkl', 'rb') as inp:
        obj = pickle.load(inp)
        print(f"> {name} loaded")
        return obj


def delete(name):
    print(f"Deleting {name}")
    os.remove(f'{pickle_path}{name}.pkl')
    print(f"> {name} deleted")

