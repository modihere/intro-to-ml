class FileInput:

	def take_input(self):
		#a try catch block to check whether the file is present or not
		try:
			with open ('aot.txt', 'r') as file:
				for lines in file:
					print(lines, end=" ")
		except FileNotFoundError as e:
			print(e)
input_file = FileInput()
input_file.take_input()