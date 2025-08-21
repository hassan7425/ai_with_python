user_input = input("Enter a number")

for i in range(1, 11):
    user_input = int(user_input)
    result = user_input * i
    print(f"{user_input} * {i} = {result}")