import MathLib

class Function:
    def __init__(self, value, name):
        self.name = name
        self.init_value = value

    def compute(self):
        pass
    
    def derivative(self):
        pass

class UniversalNum:
    @staticmethod
    def __aply__(func_name, value, **kwargs):
        try:
            return getattr(value, "__{0}__".format(func_name))(**kwargs)
        except AttributeError:
            return getattr(MathLib, str(func_name))(value, **kwargs)
    
    @staticmethod
    def sin(value):
        return UniversalNum.__aply__("sin", value)
    
    @staticmethod
    def cos(value):
        return UniversalNum.__aply__("cos", value)
    
    @staticmethod
    def tan(value):
        return UniversalNum.__aply__("tan", value)
    
    @staticmethod
    def exp(value):
        return UniversalNum.__aply__("exp", value)
    
    @staticmethod
    def log(value, base=10):
        return UniversalNum.__aply__("log", value, base)
    
    @staticmethod
    def dsin(value):
        return UniversalNum.__aply__("cos", value)

    @staticmethod
    def dcos(value):
        return -UniversalNum.__aply__("sin", value)
