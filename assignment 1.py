def split_text(string, divider):
    result = []
    temp = ""
    for c in string:
        if c == divider:
            result.append(temp)
            temp = ""
        else:
            temp += c
    if temp != "":
        result.append(temp)
    return result

def join_text(data, divider):
    out = ""
    for j in range(len(data)):
        out += data[j]
        if j != len(data) - 1:
            out += divider
    return out

sentence = input("Enter a sentence: ")
pieces = split_text(sentence, " ")

print(join_text(pieces, ","))
for item in pieces:
    print(item)
