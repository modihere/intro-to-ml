#creating a custom exception to check whether the person will be allowed in mtech or not

class AgeException(Exception):
    pass

def input_age(age):
    try:
        if (int(age)<25):
            raise AgeException
    except ValueError as e:
        return(e)
    else:
        return ("Can be admitted")
print(input_age('18'))
print(input_age('26'))

