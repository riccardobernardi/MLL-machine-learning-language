# def bind(instance, func, as_name=None):
#     """
#     Bind the function *func* to *instance*, with either provided name *as_name*
#     or the existing name of *func*. The provided *func* should accept the
#     instance as the first argument, i.e. "self".
#     """
#     if as_name is None:
#         as_name = func.__name__
#     bound_method = func.__get__(instance, instance.__class__)
#     setattr(instance, as_name, bound_method)
#     return bound_method

# class Thing:
#     def __init__(self, val):
#         self.val = val
#
# something = Thing(21)
#
# def double(self):
#     return 2 * self.val
#
# bind(something, double)
# something.double()  # returns 42

class Arr(list):
    def __init__(self, a: []):
        if not isinstance(a, list):
            a = [a]
        super().__init__(a)

    def map(self, f) -> []:
        b = []
        for i in self:
            b.append(f(i))
        return Arr(b)

    def __red__(self, f):
        if len(self) == 1:
            return self[0]
        else:
            n = f(self[0], self[1])
            self.pop(0)
            self.pop(0)
            self.insert(0, n)
            return self.__red__(f)

    def reduce(self, f):
        return Arr(self.__red__(f))

    def print(self):
        print(self)
        return Arr(self)

    def __getitem__(self, i):
        return Arr(list.__getitem__(self, i))

    def ext(self, b: []):
        return Arr(self + b)


a = Arr([1, 2, 3, 4])
print(a)
a.map(lambda x: 2 * x).print().print().reduce(lambda x, y: x + y).print()
