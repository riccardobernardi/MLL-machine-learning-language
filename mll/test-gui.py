from mll.mlltranspiler import MLL

inc = """
        x : conv2d 

        """
mll = MLL(inc, {})
mll.start()
print(mll.get_string())
print(mll.models.keys())
mll.execute()
x = mll.last_model()

print(x)