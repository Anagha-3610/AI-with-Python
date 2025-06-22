x = 10   # Integer
y = 3.14        # Float
name = "AI"     # String
is_student = True   # Boolean

print(x)
print(y)
print(name)
print(is_student)

def square(n):
    return n * n

print(square(5))


numbers = [1, 2, 3, 4, 5]
print(numbers[2])  # Output: 3
numbers.append(6)
numbers.append(7)
print(numbers)     # Output: [1, 2, 3, 4, 5, 6]
# Dictionaries

student = {"name": "Alice", "age": 21, "grade": "A"}
print(student["name"])  # Output: Alice


for num in numbers:
  print(num)
for i in range(1, 11, 1):
  print(i)

count = 0
while count < 3:
    print(count)
    count += 1
