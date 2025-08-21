
while True:

    first_number = int(input("Enter the first number: "))
    second_number = int(input("Enter the second number: "))

    def add(x,y):
        return x + y

    def subtract(x,y):
        return x - y

    def multiply(x,y):
        return x * y

    def divide(x,y):
        if y == 0:
            return "Cannot divide by zero"
        return x / y
    def exit_program():
        print("Exiting the program.")
    
    print("Select an operation:")
    print("1. Add, 2.subtract, 3.multiply, 4.divide, 5.exit")
    userOperation = input("Enter the operation (1-5): ")
    if userOperation == '1':
        result = add(first_number, second_number)
        print(f"The result of addition is: {result}")
    
    elif userOperation == '2':  
        result = subtract(first_number, second_number)
        print(f"The result of subtraction is: {result}")
    
    elif userOperation == '3':
        result = multiply(first_number, second_number)
        print(f"The result of multiplication is: {result}")

    elif userOperation == '4':
        result = divide(first_number, second_number)
        print(f"The result of division is: {result}")

    elif userOperation == '5':
        exit_program()
        break

    else:
        print("Invalid operation. Please try again.")


