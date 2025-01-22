
def convert_pkl(old_pkl, new_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    import dill
    import pickle

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl', 'data/smpl/SMPL_NEUTRAL.pkl')
