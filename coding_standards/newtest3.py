class DoOperation:
# a class for doing basic mathematical operations	

	def square_it(self, element):
		# using the ** operator to square the number
		ans = element**2
		return ans

	def division(self, first, second):
		# putting a try catch block to check division by zero
		try:
			third = first/second
			return third
		except ZeroDivisionError as e:
			return e

	def multiplication(self, one, two):
		three = one*two
		return three

	def output_of_formula(self, first):
		# declaring the coefficients of ax**2+bx
		a,b = 2,3
		output = a*(self.square_it(first)**2) + self.multiplication(b,first)
		return output

operate = DoOperation()
num_1,num_2 = map(int,input("Enter the numbers\n").split())
print("Square of the numbers are-", operate.square_it(num_1),operate.square_it(num_2))
print("division of first with second is-", operate.division(num_1, num_2))
print("Multiplication of two numbers are-", operate.multiplication(num_1, num_2))
print("Putting the first number in the formula ax**2+bx with a = 2, b = 3 we get-", operate.output_of_formula(num_1))