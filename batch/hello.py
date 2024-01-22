# Testing 'Hello World' file
import datetime

# Get the current time
current_time = datetime.datetime.now()

# Define the message to be printed and saved, including the time
message = f"Hello, this is a test message for the text file. Current time: {current_time}\n"

# Print the message to the console
print(message)

# Write (append) the message to a text file
with open("output.txt", "a") as file:
    file.write(message)

print("The message has been appended to 'output.txt'")
