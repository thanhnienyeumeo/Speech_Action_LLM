import random

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

# Khởi tạo mảng các số nguyên tố từ 11 đến 100
prime_numbers = [num for num in range(11, 101) if is_prime(num)]

# Tạo mảng mới với mỗi phần tử i là số random từ 0 đến prime_numbers[i]
random_array = [random.randint(1, prime_numbers[i]-1) for i in range(len(prime_numbers))]

print("Mảng số nguyên tố:", prime_numbers)
print("Mảng ngẫu nhiên:", random_array)
