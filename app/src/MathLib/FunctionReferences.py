
#%%

#%%


class FunctionRef:
    refs = {}

    @staticmethod
    def add(class_obj):
        if class_obj.__name__ not in FunctionRef.refs :
            # thread safe since a particular instance is not the target, just the class object without its parameters

            FunctionRef.refs[class_obj.__name__] = class_obj # in order to retrieve class by name even if the function has not been defined in that module
        
    @staticmethod
    def get(ref):
        return FunctionRef.refs[ref]
    
    @staticmethod
    def getNew(ref):
        return FunctionRef.refs[ref]()

