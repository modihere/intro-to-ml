class add:

	def __init__(self,First,Second):
		
		self.first = First
		self.second = Second
		self.output = 0

	def adder(self):
		
		self.output = self.first+self.second
		return self.output

obj = add(2,3)
Sum = obj.adder()
print(Sum)