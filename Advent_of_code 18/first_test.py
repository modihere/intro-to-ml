# question 2a


def do_it():
    twos, threes = 0, 0
    with open("aoc1.txt") as file:
        for line in file:
            char_list = [char for char in line]
            count_list = [0 for i in range(255)]
            for char in char_list:
                count_list[ord(char)] += 1
            for count in count_list:
                if count == 2:
                    some_two = 1
                    twos = twos + some_two
                    break
            for count in count_list:
                if count == 3:
                    some_three = 1
                    threes = threes + some_three
                    break
        print (twos * threes)


do_it()

# question 2b


with open('aoc1.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]


def check_one(string1, string2):
    if len(string1) == len(string2):
        count_diffs = 0
        for a, b in zip(string1, string2):
            if a != b:
                if count_diffs: return False
                count_diffs += 1
        return True


for i in range(len(content)):
    for j in range(len(content)):
        if (content[i] != content[j]):
            check = check_one(content[i], content[j])
            if (check == True):
                str1 = content[i]
                str2 = content[j]
                break
    if (check == True):
        break
# print(str1)
# print(str2)
for i in range(len(str1)):
    if (str1[i] != str2[i]):
        break
newstring = str1[:i] + str2[i + 1:]
print(newstring)


# question 5a


def do_it():
    with open("aoc2.txt") as file:
        for line in file:
            char_list = [char for char in line]
            # print(len(char_list))
            i = 0
            while i < (len(char_list) - 1):

                if ord(char_list[i]) == ord(char_list[i + 1]) - 32:
                    # print(i,i+1)
                    char_list = char_list[:i] + char_list[i + 2:]
                    i = i - 1
                    # print(len(char_list))
                elif ord(char_list[i]) == ord(char_list[i + 1]) + 32:
                    # print(i,i+1)
                    char_list = char_list[:i] + char_list[i + 2:]
                    i = i - 1
                    # print(len(char_list))
                else:
                    i = i + 1
            print(len(char_list))


do_it()

# question 5b


with open('aoc2.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]


def less_optimized(string):
    char_list = [char for char in string]
    # print(len(char_list))
    i = 0
    while i < (len(char_list) - 1):

        if ord(char_list[i]) == ord(char_list[i + 1]) - 32:
            # print(i,i+1)
            char_list = char_list[:i] + char_list[i + 2:]
            i = i - 1
            # print(len(char_list))
        elif ord(char_list[i]) == ord(char_list[i + 1]) + 32:
            # print(i,i+1)
            char_list = char_list[:i] + char_list[i + 2:]
            i = i - 1
            # print(len(char_list))
        else:
            i = i + 1
    return (len(char_list))


def more_optimized(string):
    l = list(string)
    i = 0
    while True:
        if (l[i].swapcase() == l[i + 1]):
            del l[i]
            del l[i]
            if (i > 0):
                i -= 1
        else:
            i += 1
        if (i >= len(l) - 1):
            break
    # print(l)
    return len(l)


m = 0
s1 = [chr(i) for i in range(ord('a'), ord('z') + 1)]
s2 = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
new_str = content[0]
m = 0
for i in range(26):
    new_str = new_str.replace(s1[i], '')
    new_str = new_str.replace(s2[i], '')
    r = more_optimized(new_str)  # call with more_optimized(new_str) for faster operation
    if (m == 0):
        m = r
    else:
        if (m > r):
            m = r
    new_str = content[0]

print(m)
