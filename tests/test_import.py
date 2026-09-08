from src.Modal_Decomposition.Base.Cache import Cache

def test_import():
    import scipy.signal as ss

    Cache.add("ss", ss)
    Cache.get("ss")

    Cache.add("ss", ss)