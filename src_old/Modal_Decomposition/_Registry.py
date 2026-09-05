_ClassRegistry = {}
_FunctionRegistry = {}

def register_class(name: str):
    def dec(cls):
        _ClassRegistry[name] = cls
        return cls
    return dec

def register_function(name: str):
    def dec(func):
        _FunctionRegistry[name] = func
        return func
    return dec