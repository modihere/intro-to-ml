class Length:

	def __init__(self, list_len):

		self.list_len = list_len

	def length_of_list(self):
		# function to check the lenght of the list using the len function.
		length = len(self.list_len)
		return length

A = [1, 2, 3, 4, 5, 6, 7]
len_list = Length(A)
print(len_list.length_of_list())
