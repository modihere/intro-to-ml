class DecoratorExample:

	def __init__(self):
		print('Hello, World!')

	@classmethod
	def example_function(cls):
		""" This method is a class method"""
		print('I\'m a class method!')
		cls.some_other_function()

	@staticmethod
	def some_other_function():
		print('Hello!')

de = DecoratorExample()
de.example_function()