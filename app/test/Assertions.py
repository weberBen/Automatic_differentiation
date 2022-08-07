import math 

#%%

class IsCloseAssertion:
    def assertIsClose(self, x, y, msg=None):

        if not math.isclose(x, y):
            msg = str(x) + "!=" + str(y) + " difference(" + ('%.15f' % abs(x-y)) + ")" + ((" : " + str(msg)) if msg is not None else "")
            raise AssertionError(msg)