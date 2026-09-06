"""
store of the import cache
"""

class cache:
    def __init__(self):
        _cache = {}

    def add(self, name, api, verbose: bool) -> bool:
        """
        add the api of the module or function or class to the cache
        :param name:
        :param api:
        :param verbose:
        :return:
        """
        ...

    def get(self, name):
        """
        return the cached api
        :param name:
        :return:
        """

    def check(self, name) -> bool:
        """
        return the result of the query to cache
        :param name:
        :return:
        """