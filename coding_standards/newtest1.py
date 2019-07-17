class FindChar:

	def __init__(self, string):
		# constructor initialization
		self.string = string

	def find_char(self, check, element):
		# function to return the second element or the element being searched
		if check == 0:
			# returning the second element in this block
			return self.string[1]
		else:
			# a exception handler to check whether the element searched is there or not.
			try:
				return self.string.index(element)
			except ValueError as e:
				return e			

if __name__ == "__main__":
	string = "aeiou"
	find_character = FindChar(string)
	check = int(input("1 - if a user defined element needs to be searched" 
						" else 0 - for showing 2nd element "))
	if check == 1:
		character = input("Enter the character ")
		print(find_character.find_char(1,character))
	else:
		print(find_character.find_char(0,None))	

