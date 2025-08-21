current_balance = 1000  # Example balance, can be modified
while True:   
    print('This is the ATM allocation system')
    print('1. Check balance, 2. Deposit money, 3. Withdraw money, 4. Exit')
    userInput = input('Please select an option (1-4): ')

    if  userInput == '1':
        print(f'Your current balance is {current_balance}')
    elif userInput == '2':
        deposit_amount = int(input('Enter the amount to deposit: '))
        current_balance += deposit_amount
        print(f'You have successfully deposited {deposit_amount}')

    elif userInput == '3':
        withdraw_amount = int(input('Enter the amount to withdraw: '))
        if withdraw_amount > current_balance:
            print('Insufficient funds for this withdrawal.')
        else:
            current_balance -= withdraw_amount
            print(f'You have successfully withdrawn {withdraw_amount}')
    elif userInput == '4':
        print('Thank you!')
        break
