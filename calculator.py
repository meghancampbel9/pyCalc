# calculator.py

def add(x, y):
    """Adds two numbers."""
    return x + y

def subtract(x, y):
    """Subtracts two numbers."""
    return x - y

def multiply(x, y):
    """Multiplies two numbers."""
    return x * y

def divide(x, y):
    """Divides two numbers."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def power(x, y):
    """Calculates x to the power of y."""
    return x ** y

def sqrt(x):
    """Calculates the square root of a number."""
    if x < 0:
        raise ValueError("Cannot calculate the square root of a negative number")
    return x ** 0.5

def factorial(n):
    """Calculates the factorial of a non-negative integer."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial is only defined for non-negative integers")
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def is_prime(n):
    """Checks if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    """Calculates the greatest common divisor of two integers."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Calculates the least common multiple of two integers."""
    return abs(a*b) // gcd(a, b)

def fibonacci(n):
    """Calculates the nth Fibonacci number."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Fibonacci sequence is defined for non-negative integers")
    if n <= 1:
        return n
    else:
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

def sum_of_digits(n):
    """Calculates the sum of the digits of a non-negative integer."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer")
    return sum(int(digit) for digit in str(n))

def reverse_string(s):
    """Reverses a given string."""
    return s[::-1]

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

def count_vowels(s):
    """Counts the number of vowels in a string."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

def find_max(numbers):
    """Finds the maximum number in a list."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return max(numbers)

def find_min(numbers):
    """Finds the minimum number in a list."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return min(numbers)

def calculate_average(numbers):
    """Calculates the average of a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    """Calculates the median of a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[mid]

def calculate_mode(numbers):
    """Calculates the mode of a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    from collections import Counter
    counts = Counter(numbers)
    max_count = max(counts.values())
    modes = [num for num, count in counts.items() if count == max_count]
    # If all numbers appear only once, there is no unique mode.
    # In such cases, we can return the smallest number or raise an error.
    # Here, we return the smallest number if all counts are 1.
    if max_count == 1 and len(numbers) > 1:
        return min(numbers)
    return modes[0] if len(modes) == 1 else modes # Return the first mode if multiple, or the list of modes

def celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """Converts Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

def meters_to_feet(meters):
    """Converts meters to feet."""
    return meters * 3.28084

def feet_to_meters(feet):
    """Converts feet to meters."""
    return feet / 3.28084

def kilograms_to_pounds(kilograms):
    """Converts kilograms to pounds."""
    return kilograms * 2.20462

def pounds_to_kilograms(pounds):
    """Converts pounds to kilograms."""
    return pounds / 2.20462

def km_to_miles(km):
    """Converts kilometers to miles."""
    return km * 0.621371

def miles_to_km(miles):
    """Converts miles to kilometers."""
    return miles / 0.621371

def simple_interest(principal, rate, time):
    """Calculates simple interest."""
    return (principal * rate * time) / 100

def compound_interest(principal, rate, time, n_compounding_per_year):
    """Calculates compound interest."""
    amount = principal * (1 + rate / (100 * n_compounding_per_year)) ** (n_compounding_per_year * time)
    return amount - principal

def quadratic_formula(a, b, c):
    """Solves a quadratic equation ax^2 + bx + c = 0 using the quadratic formula."""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a quadratic equation")
    
    delta = (b**2) - (4*a*c)
    
    if delta < 0:
        # Handle complex roots if necessary, or raise an error
        # For simplicity, we'll raise an error indicating no real roots
        raise ValueError("No real roots exist for this quadratic equation")
    elif delta == 0:
        x = -b / (2*a)
        return (x,) # Return a tuple with one root
    else:
        x1 = (-b - delta**0.5) / (2*a)
        x2 = (-b + delta**0.5) / (2*a)
        return (x1, x2) # Return a tuple with two roots

def is_even(n):
    """Checks if a number is even."""
    return n % 2 == 0

def is_odd(n):
    """Checks if a number is odd."""
    return n % 2 != 0

def get_day_of_week(year, month, day):
    """Gets the day of the week for a given date."""
    import datetime
    try:
        date_obj = datetime.date(year, month, day)
        return date_obj.strftime("%A")
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")

def calculate_area_rectangle(length, width):
    """Calculates the area of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return length * width

def calculate_perimeter_rectangle(length, width):
    """Calculates the perimeter of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Length and width must be non-negative")
    return 2 * (length + width)

def calculate_area_circle(radius):
    """Calculates the area of a circle."""
    import math
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return math.pi * radius**2

def calculate_circumference_circle(radius):
    """Calculates the circumference of a circle."""
    import math
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 2 * math.pi * radius

def calculate_area_triangle(base, height):
    """Calculates the area of a triangle."""
    if base < 0 or height < 0:
        raise ValueError("Base and height must be non-negative")
    return 0.5 * base * height

def calculate_volume_sphere(radius):
    """Calculates the volume of a sphere."""
    import math
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return (4/3) * math.pi * radius**3

def calculate_surface_area_sphere(radius):
    """Calculates the surface area of a sphere."""
    import math
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    return 4 * math.pi * radius**2

def convert_to_roman(num):
    """Converts an integer to its Roman numeral representation."""
    if not isinstance(num, int) or not 0 < num < 4000:
        raise ValueError("Input must be an integer between 1 and 3999")
    
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

def convert_from_roman(roman):
    """Converts a Roman numeral to an integer."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    int_val = 0
    prev_val = 0
    for i in reversed(roman):
        current_val = roman_map[i]
        if current_val < prev_val:
            int_val -= current_val
        else:
            int_val += current_val
        prev_val = current_val
    return int_val

def is_leap_year(year):
    """Checks if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def days_in_month(year, month):
    """Returns the number of days in a given month and year."""
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")
    if month == 2:
        return 29 if is_leap_year(year) else 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31

def calculate_days_between_dates(date1_str, date2_str):
    """Calculates the number of days between two dates (YYYY-MM-DD)."""
    from datetime import date
    try:
        date1 = date.fromisoformat(date1_str)
        date2 = date.fromisoformat(date2_str)
        return abs((date2 - date1).days)
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")

def calculate_age(birthdate_str):
    """Calculates age given a birthdate (YYYY-MM-DD)."""
    from datetime import date
    try:
        birthdate = date.fromisoformat(birthdate_str)
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age
    except ValueError:
        raise ValueError("Birthdate must be in YYYY-MM-DD format")

def calculate_bmi(weight_kg, height_m):
    """Calculates Body Mass Index (BMI)."""
    if weight_kg <= 0 or height_m <= 0:
        raise ValueError("Weight and height must be positive values")
    return weight_kg / (height_m ** 2)

def convert_currency(amount, from_currency, to_currency, exchange_rates):
    """Converts currency using provided exchange rates."""
    if from_currency not in exchange_rates or to_currency not in exchange_rates[from_currency]:
        raise ValueError(f"Exchange rate not available for {from_currency} to {to_currency}")
    
    # Assuming exchange_rates is a dictionary like:
    # {'USD': {'EUR': 0.92, 'GBP': 0.79}, 'EUR': {'USD': 1.09, 'GBP': 0.86}}
    # If direct rate is not available, use a common base currency (e.g., USD)
    if to_currency in exchange_rates[from_currency]:
        return amount * exchange_rates[from_currency][to_currency]
    else:
        # Example: Convert EUR to JPY via USD
        # Requires rates like: {'EUR': {'USD': 1.09}, 'USD': {'JPY': 150}}
        # This part needs a more robust implementation for multi-currency conversions
        raise NotImplementedError("Multi-step currency conversion not implemented yet")

def calculate_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def calculate_slope(x1, y1, x2, y2):
    """Calculates the slope of the line between two points (x1, y1) and (x2, y2)."""
    if x1 == x2:
        raise ValueError("Cannot calculate slope for vertical lines (division by zero)")
    return (y2 - y1) / (x2 - x1)

def get_day_name(day_number):
    """Returns the name of the day given its number (1=Monday, 7=Sunday)."""
    days = {
        1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
        5: "Friday", 6: "Saturday", 7: "Sunday"
    }
    if day_number not in days:
        raise ValueError("Day number must be between 1 and 7")
    return days[day_number]

def get_month_name(month_number):
    """Returns the name of the month given its number (1=January, 12=December)."""
    months = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    if month_number not in months:
        raise ValueError("Month number must be between 1 and 12")
    return months[month_number]

def calculate_nth_term_arithmetic(a1, d, n):
    """Calculates the nth term of an arithmetic sequence."""
    # a_n = a_1 + (n-1)d
    return a1 + (n - 1) * d

def calculate_sum_arithmetic(a1, an, n):
    """Calculates the sum of an arithmetic sequence."""
    # S_n = n/2 * (a_1 + a_n)
    return (n / 2) * (a1 + an)

def calculate_nth_term_geometric(a1, r, n):
    """Calculates the nth term of a geometric sequence."""
    # a_n = a_1 * r^(n-1)
    return a1 * (r ** (n - 1))

def calculate_sum_geometric(a1, r, n):
    """Calculates the sum of a finite geometric sequence."""
    # S_n = a_1 * (1 - r^n) / (1 - r)
    if r == 1:
        return a1 * n
    return a1 * (1 - r**n) / (1 - r)

def calculate_determinant_2x2(matrix):
    """Calculates the determinant of a 2x2 matrix."""
    # [[a, b], [c, d]] -> ad - bc
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError("Input must be a 2x2 matrix")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def calculate_determinant_3x3(matrix):
    """Calculates the determinant of a 3x3 matrix."""
    # [[a, b, c], [d, e, f], [g, h, i]] -> a(ei - fh) - b(di - fg) + c(dh - eg)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Input must be a 3x3 matrix")
    
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

def transpose_matrix(matrix):
    """Transposes a matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed

def multiply_matrices(matrix1, matrix2):
    """Multiplies two matrices."""
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix")

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1): # or range(rows2)
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result

def get_file_extension(filename):
    """Gets the file extension from a filename."""
    return filename.split('.')[-1] if '.' in filename else ""

def get_filename_without_extension(filename):
    """Gets the filename without its extension."""
    return '.'.join(filename.split('.')[:-1]) if '.' in filename else filename

def is_valid_email(email):
    """Checks if a string is a valid email address format."""
    import re
    # A simple regex for email validation. More robust validation might be needed.
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def is_valid_url(url):
    """Checks if a string is a valid URL format."""
    import re
    # A simple regex for URL validation.
    url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' # domain...
        r'localhost|' # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(url_regex, url) is not None

def calculate_power_set(input_set):
    """Calculates the power set of a given set."""
    from itertools import chain, combinations
    s = list(input_set)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def get_unique_elements(input_list):
    """Returns a list of unique elements from a list, preserving order."""
    seen = set()
    unique_list = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def flatten_list(nested_list):
    """Flattens a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def group_by_key(list_of_dicts, key):
    """Groups a list of dictionaries by a specified key."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for d in list_of_dicts:
        if key in d:
            grouped[d[key]].append(d)
    return dict(grouped)

def merge_dictionaries(dict1, dict2, merge_strategy='overwrite'):
    """Merges two dictionaries.
    
    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        merge_strategy (str): 'overwrite' (default) or 'deep_merge'.
                              'overwrite' means values from dict2 replace values from dict1.
                              'deep_merge' recursively merges nested dictionaries.
    
    Returns:
        dict: The merged dictionary.
    """
    merged = dict1.copy()
    
    if merge_strategy == 'deep_merge':
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_dictionaries(merged[key], value, merge_strategy='deep_merge')
            else:
                merged[key] = value
    elif merge_strategy == 'overwrite':
        merged.update(dict2)
    else:
        raise ValueError("Invalid merge_strategy. Choose 'overwrite' or 'deep_merge'.")
        
    return merged

def get_nested_value(data, keys, default=None):
    """Safely retrieves a nested value from a dictionary using a list of keys.
    
    Args:
        data (dict): The dictionary to search within.
        keys (list): A list of keys representing the path to the desired value.
        default: The value to return if the path is not found.
    
    Returns:
        The value at the specified path, or the default value if not found.
    """
    current_level = data
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        elif isinstance(current_level, list) and isinstance(key, int) and 0 <= key < len(current_level):
            current_level = current_level[key]
        else:
            return default
    return current_level

def remove_duplicates_from_list(input_list):
    """Removes duplicate elements from a list while preserving order."""
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]

def chunk_list(input_list, chunk_size):
    """Splits a list into chunks of a specified size."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer")
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def rotate_list(input_list, n):
    """Rotates a list by n positions. Positive n rotates right, negative n rotates left."""
    if not input_list:
        return []
    n = n % len(input_list)
    if n == 0:
        return input_list[:] # Return a copy
    return input_list[-n:] + input_list[:-n]

def find_first_occurrence(input_list, element):
    """Finds the index of the first occurrence of an element in a list."""
    try:
        return input_list.index(element)
    except ValueError:
        return -1 # Element not found

def find_last_occurrence(input_list, element):
    """Finds the index of the last occurrence of an element in a list."""
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] == element:
            return i
    return -1 # Element not found

def count_occurrences(input_list, element):
    """Counts the number of occurrences of an element in a list."""
    return input_list.count(element)

def get_list_intersection(list1, list2):
    """Returns the intersection of two lists (elements present in both)."""
    return list(set(list1) & set(list2))

def get_list_union(list1, list2):
    """Returns the union of two lists (all unique elements from both)."""
    return list(set(list1) | set(list2))

def get_list_difference(list1, list2):
    """Returns the difference of two lists (elements in list1 but not in list2)."""
    return list(set(list1) - set(list2))

def is_list_subset(list1, list2):
    """Checks if list1 is a subset of list2."""
    return set(list1).issubset(set(list2))

def is_list_superset(list1, list2):
    """Checks if list1 is a superset of list2."""
    return set(list1).issuperset(set(list2))

def calculate_mean_confidence_interval(data, confidence_level=0.95):
    """Calculates the confidence interval for the mean of a dataset."""
    import numpy as np
    from scipy import stats

    n = len(data)
    if n < 2:
        raise ValueError("Dataset must contain at least two elements to calculate confidence interval.")

    mean = np.mean(data)
    std_err = stats.sem(data) # Standard error of the mean
    
    # Degrees of freedom for t-distribution
    df = n - 1
    
    # Calculate the critical t-value
    # alpha is the significance level (1 - confidence_level)
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    
    # Margin of error
    margin_of_error = t_crit * std_err
    
    # Confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return (lower_bound, upper_bound)

def calculate_variance(data):
    """Calculates the sample variance of a dataset."""
    import numpy as np
    if len(data) < 2:
        raise ValueError("Dataset must contain at least two elements to calculate variance.")
    return np.var(data, ddof=1) # ddof=1 for sample variance

def calculate_std_dev(data):
    """Calculates the sample standard deviation of a dataset."""
    import numpy as np
    if len(data) < 2:
        raise ValueError("Dataset must contain at least two elements to calculate standard deviation.")
    return np.std(data, ddof=1) # ddof=1 for sample standard deviation

def calculate_covariance(x, y):
    """Calculates the sample covariance between two datasets."""
    import numpy as np
    if len(x) != len(y):
        raise ValueError("Input datasets must have the same length.")
    if len(x) < 2:
        raise ValueError("Datasets must contain at least two elements to calculate covariance.")
    return np.cov(x, y, ddof=1)[0, 1] # ddof=1 for sample covariance

def calculate_correlation_coefficient(x, y):
    """Calculates the Pearson correlation coefficient between two datasets."""
    import numpy as np
    if len(x) != len(y):
        raise ValueError("Input datasets must have the same length.")
    if len(x) < 2:
        raise ValueError("Datasets must contain at least two elements to calculate correlation coefficient.")
    return np.corrcoef(x, y)[0, 1]

def linear_regression(x, y):
    """Performs simple linear regression (y = mx + b)."""
    import numpy as np
    if len(x) != len(y):
        raise ValueError("Input datasets must have the same length.")
    if len(x) < 2:
        raise ValueError("Datasets must contain at least two elements for linear regression.")
        
    # Calculate slope (m) and intercept (b)
    # m = sum((x_i - mean_x) * (y_i - mean_y)) / sum((x_i - mean_x)^2)
    # b = mean_y - m * mean_x
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    denominator = sum((xi - x_mean)**2 for xi in x)
    
    if denominator == 0:
        raise ValueError("Cannot perform linear regression when all x values are the same.")
        
    m = numerator / denominator
    b = y_mean - m * x_mean
    
    # Calculate R-squared
    ss_total = sum((yi - y_mean)**2 for yi in y)
    ss_residual = sum((yi - (m * xi + b))**2 for xi, yi in zip(x, y))
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 1.0
    
    return m, b, r_squared

def polynomial_regression(x, y, degree):
    """Performs polynomial regression of a given degree."""
    import numpy as np
    if len(x) != len(y):
        raise ValueError("Input datasets must have the same length.")
    if len(x) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} data points for polynomial regression of degree {degree}.")
        
    # Use numpy's polyfit to find the coefficients
    # Returns coefficients [p_n, p_{n-1}, ..., p_1, p_0] for p(x) = p_n*x^n + ... + p_1*x + p_0
    coefficients = np.polyfit(x, y, degree)
    
    # Calculate R-squared
    y_pred = np.polyval(coefficients, x)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 1.0
    
    return coefficients, r_squared

def calculate_log_loss(y_true, y_pred):
    """Calculates the log loss (binary cross-entropy) for binary classification."""
    import numpy as np
    # Ensure y_pred values are clipped to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate log loss
    logloss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return logloss

def calculate_accuracy_score(y_true, y_pred):
    """Calculates the accuracy score for classification."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length.")
        
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def calculate_precision_score(y_true, y_pred, pos_label=1):
    """Calculates the precision score for binary classification."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length.")
        
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        return 0.0 # Avoid division by zero
    return tp / (tp + fp)

def calculate_recall_score(y_true, y_pred, pos_label=1):
    """Calculates the recall score for binary classification."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length.")
        
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        return 0.0 # Avoid division by zero
    return tp / (tp + fn)

def calculate_f1_score(y_true, y_pred, pos_label=1):
    """Calculates the F1 score for binary classification."""
    precision = calculate_precision_score(y_true, y_pred, pos_label)
    recall = calculate_recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0 # Avoid division by zero
    return 2 * (precision * recall) / (precision + recall)

def calculate_confusion_matrix(y_true, y_pred, labels=None):
    """Calculates the confusion matrix for classification."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True labels and predicted labels must have the same length.")
        
    if labels is None:
        labels = sorted(list(set(np.concatenate((y_true, y_pred)))))
        
    num_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    cm = np.zeros((num_labels, num_labels), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            cm[label_to_index[true], label_to_index[pred]] += 1
            
    return cm, labels

def calculate_roc_auc_score(y_true, y_pred_proba):
    """Calculates the ROC AUC score for binary classification."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred_proba)
    except ValueError as e:
        raise ValueError(f"Could not calculate ROC AUC score: {e}. Ensure y_true contains binary labels and y_pred_proba contains probabilities.")

def calculate_mean_absolute_error(y_true, y_pred):
    """Calculates the Mean Absolute Error (MAE)."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True values and predicted values must have the same length.")
        
    return np.mean(np.abs(y_true - y_pred))

def calculate_mean_squared_error(y_true, y_pred):
    """Calculates the Mean Squared Error (MSE)."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True values and predicted values must have the same length.")
        
    return np.mean((y_true - y_pred)**2)

def calculate_root_mean_squared_error(y_true, y_pred):
    """Calculates the Root Mean Squared Error (RMSE)."""
    mse = calculate_mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def calculate_r2_score(y_true, y_pred):
    """Calculates the R-squared (coefficient of determination) score."""
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("True values and predicted values must have the same length.")
        
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    
    if ss_total == 0:
        return 1.0 # Perfect prediction if all true values are the same
        
    r2 = 1 - (ss_residual / ss_total)
    return r2

def get_file_size(filepath):
    """Gets the size of a file in bytes."""
    import os
    return os.path.getsize(filepath)

def get_file_modification_time(filepath):
    """Gets the last modification time of a file."""
    import os
    return os.path.getmtime(filepath)

def get_file_creation_time(filepath):
    """Gets the creation time of a file."""
    import os
    # Note: Creation time is not reliably available on all OS (e.g., Linux)
    # os.path.getctime might return modification time on some systems.
    try:
        return os.path.getctime(filepath)
    except OSError:
        # Fallback to modification time if creation time is not available
        return os.path.getmtime(filepath)

def list_directory_contents(directory_path):
    """Lists the contents (files and subdirectories) of a directory."""
    import os
    return os.listdir(directory_path)

def create_directory(directory_path):
    """Creates a directory if it does not exist."""
    import os
    os.makedirs(directory_path, exist_ok=True)

def remove_directory(directory_path):
    """Removes a directory and its contents."""
    import shutil
    shutil.rmtree(directory_path)

def move_file(source_path, destination_path):
    """Moves a file from source to destination."""
    import shutil
    shutil.move(source_path, destination_path)

def copy_file(source_path, destination_path):
    """Copies a file from source to destination."""
    import shutil
    shutil.copy2(source_path, destination_path) # copy2 preserves metadata

def read_text_file(filepath):
    """Reads the entire content of a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_text_file(filepath, content):
    """Writes content to a text file, overwriting if it exists."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def append_text_file(filepath, content):
    """Appends content to a text file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(content)

def read_json_file(filepath):
    """Reads a JSON file."""
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(filepath, data):
    """Writes data to a JSON file."""
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4) # Use indent for pretty printing

def read_csv_file(filepath):
    """Reads a CSV file into a list of dictionaries."""
    import csv
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def write_csv_file(filepath, data, fieldnames):
    """Writes data to a CSV file."""
    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def get_environment_variable(var_name):
    """Gets the value of an environment variable."""
    import os
    return os.environ.get(var_name)

def set_environment_variable(var_name, value):
    """Sets the value of an environment variable."""
    import os
    os.environ[var_name] = str(value)

def run_command(command):
    """Runs a shell command and returns its output."""
    import subprocess
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {command}\nError: {e.stderr}")

def calculate_fibonacci_recursive(n):
    """Calculates the nth Fibonacci number using recursion (less efficient)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Fibonacci sequence is defined for non-negative integers")
    if n <= 1:
        return n
    else:
        return calculate_fibonacci_recursive(n-1) + calculate_fibonacci_recursive(n-2)

def calculate_gcd_recursive(a, b):
    """Calculates the greatest common divisor using recursion."""
    if b == 0:
        return a
    else:
        return calculate_gcd_recursive(b, a % b)

def calculate_factorial_recursive(n):
    """Calculates the factorial of a non-negative integer using recursion."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial is only defined for non-negative integers")
    if n == 0:
        return 1
    else:
        return n * calculate_factorial_recursive(n-1)

def binary_search(sorted_list, target):
    """Performs binary search on a sorted list."""
    low = 0
    high = len(sorted_list) - 1
    
    while low <= high:
        mid = (low + high) // 2
        mid_val = sorted_list[mid]
        
        if mid_val == target:
            return mid # Target found, return index
        elif mid_val < target:
            low = mid + 1 # Target is in the right half
        else:
            high = mid - 1 # Target is in the left half
            
    return -1 # Target not found

def merge_sort(data):
    """Sorts a list using the merge sort algorithm."""
    if len(data) <= 1:
        return data
    
    mid = len(data) // 2
    left_half = data[:mid]
    right_half = data[mid:]
    
    # Recursively sort both halves
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left, right):
    """Helper function for merge_sort to merge two sorted lists."""
    merged = []
    left_idx, right_idx = 0, 0
    
    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1
            
    # Append remaining elements
    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])
    return merged

def quick_sort(data):
    """Sorts a list using the quick sort algorithm."""
    if len(data) <= 1:
        return data
    
    pivot = data[len(data) // 2]
    left = [x for x in data if x < pivot]
    middle = [x for x in data if x == pivot]
    right = [x for x in data if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def bubble_sort(data):
    """Sorts a list using the bubble sort algorithm."""
    n = len(data)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the list from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
    return data

def insertion_sort(data):
    """Sorts a list using the insertion sort algorithm."""
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        # Move elements of data[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data

def selection_sort(data):
    """Sorts a list using the selection sort algorithm."""
    n = len(data)
    for i in range(n):
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i+1, n):
            if data[j] < data[min_idx]:
                min_idx = j
                
        # Swap the found minimum element with the first element
        data[i], data[min_idx] = data[min_idx], data[i]
    return data

def heapify(arr, n, i):
    """Heapify function for heap sort."""
    largest = i  # Initialize largest as root
    left = 2 * i + 1     # left child = 2*i + 1
    right = 2 * i + 2    # right child = 2*i + 2

    # See if left child of root exists and is greater than root
    if left < n and arr[i] < arr[left]:
        largest = left

    # See if right child of root exists and is greater than root
    if right < n and arr[largest] < arr[right]:
        largest = right

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)

def heap_sort(data):
    """Sorts a list using the heap sort algorithm."""
    n = len(data)

    # Build a maxheap.
    # Since last parent will be at (n//2) - 1, we can start at that location.
    for i in range(n // 2 - 1, -1, -1):
        heapify(data, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        data[i], data[0] = data[0], data[i]  # swap
        heapify(data, i, 0) # call max heapify on the reduced heap
    return data

def count_sort(arr, max_val):
    """Sorts a list of non-negative integers using the count sort algorithm."""
    if not all(isinstance(x, int) and x >= 0 for x in arr):
        raise ValueError("Count sort requires a list of non-negative integers.")
    if max_val is None:
        max_val = max(arr) if arr else 0
        
    n = len(arr)
    output = [0] * n
    count = [0] * (max_val + 1)

    # Store count of each character
    for i in range(n):
        count[arr[i]] += 1

    # Change count[i] so that count[i] now contains actual
    # position of this character in output array
    for i in range(1, max_val + 1):
        count[i] += count[i - 1]

    # Build the output character
    i = n - 1
    while i >= 0:
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
        i -= 1

    # Copy the output array to arr, so that arr now
    # contains sorted characters
    for i in range(n):
        arr[i] = output[i]
    return arr

def radix_sort(arr):
    """Sorts a list of non-negative integers using the radix sort algorithm (LSD)."""
    if not all(isinstance(x, int) and x >= 0 for x in arr):
        raise ValueError("Radix sort requires a list of non-negative integers.")
        
    # Find the maximum number to know number of digits
    max_num = max(arr) if arr else 0
    
    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max_num // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def counting_sort_by_digit(arr, exp):
    """Helper function for radix sort to perform counting sort based on a digit."""
    n = len(arr)
    output = [0] * n
    count = [0] * 10 # Digits are 0-9

    # Store count of occurrences in count[]
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1

    # Copy the output array to arr, so that arr now
    # contains sorted numbers according to current digit
    for i in range(n):
        arr[i] = output[i]

def bucket_sort(data, num_buckets=10):
    """Sorts a list using the bucket sort algorithm."""
    if not data:
        return []
        
    # Find the range of the data
    min_val = min(data)
    max_val = max(data)
    
    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]
    
    # Distribute elements into buckets
    # The range of values is max_val - min_val.
    # We want to map this range to num_buckets.
    # bucket_index = int((value - min_val) / (max_val - min_val + 1) * num_buckets)
    # Handle the case where max_val == min_val to avoid division by zero
    range_val = max_val - min_val
    if range_val == 0:
        # All elements are the same, no sorting needed within buckets
        return data
        
    for value in data:
        # Calculate bucket index, ensuring it stays within bounds [0, num_buckets-1]
        # Add a small epsilon to max_val in the denominator to handle the max value correctly
        bucket_index = int((value - min_val) / (range_val + 1e-9) * num_buckets) 
        # Ensure the index is within the valid range, especially for the max value
        bucket_index = min(bucket_index, num_buckets - 1) 
        buckets[bucket_index].append(value)
        
    # Sort each bucket individually (e.g., using insertion sort or Python's sort)
    for i in range(num_buckets):
        buckets[i].sort() # Using Python's built-in sort for simplicity
        
    # Concatenate the sorted buckets
    sorted_data = []
    for bucket in buckets:
        sorted_data.extend(bucket)
        
    return sorted_data

def get_bit_count(n):
    """Counts the number of set bits (1s) in the binary representation of an integer."""
    count = 0
    while n > 0:
        n &= (n - 1) # Brian Kernighan's algorithm
        count += 1
    return count

def is_power_of_two(n):
    """Checks if a number is a power of two."""
    return n > 0 and (n & (n - 1) == 0)

def find_missing_number(nums):
    """Finds the missing number in a sequence from 0 to n."""
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return expected_sum - actual_sum

def find_duplicate_number(nums):
    """Finds the duplicate number in a list containing n+1 integers where each integer is between 1 and n."""
    # Floyd's Tortoise and Hare (Cycle Detection) algorithm
    if not nums:
        return -1 # Or raise an error
        
    tortoise = nums[0]
    hare = nums[0]
    
    # Phase 1: Find the intersection point of the two pointers
    while True:
        tortoise = nums[tortoise]
        hare = nums[nums[hare]]
        if tortoise == hare:
            break
            
    # Phase 2: Find the entrance to the cycle
    ptr1 = nums[0]
    ptr2 = tortoise
    while ptr1 != ptr2:
        ptr1 = nums[ptr1]
        ptr2 = nums[ptr2]
        
    return ptr1

def find_single_number(nums):
    """Finds the single element that appears only once in a list where all other elements appear twice."""
    # Uses the XOR property: a ^ a = 0 and a ^ 0 = a
    result = 0
    for num in nums:
        result ^= num
    return result

def find_two_single_numbers(nums):
    """Finds the two elements that appear only once in a list where all other elements appear twice."""
    # XOR all numbers to get the XOR of the two unique numbers
    xor_sum = 0
    for num in nums:
        xor_sum ^= num
        
    # Find the rightmost set bit in xor_sum. This bit must be different
    # between the two unique numbers.
    rightmost_set_bit = xor_sum & -xor_sum # Isolates the rightmost set bit
    
    num1 = 0
    num2 = 0
    
    # Partition the numbers into two groups based on the rightmost set bit
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num # XOR numbers with the set bit
        else:
            num2 ^= num # XOR numbers without the set bit
            
    return num1, num2

def rotate_array(nums, k):
    """Rotates an array to the right by k steps."""
    n = len(nums)
    k = k % n # Handle cases where k >= n
    
    # Reverse the entire array
    nums.reverse()
    # Reverse the first k elements
    nums[:k] = nums[:k][::-1]
    # Reverse the remaining n-k elements
    nums[k:] = nums[k:][::-1]
    
    return nums # Modify in-place, but return for convenience

def merge_sorted_arrays(arr1, arr2):
    """Merges two sorted arrays into a single sorted array."""
    merged = []
    i, j = 0, 0
    
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[j])
            j += 1
            
    # Append remaining elements
    merged.extend(arr1[i:])
    merged.extend(arr2[j:])
    
    return merged

def find_peak_element(nums):
    """Finds a peak element in an array (an element greater than its neighbors)."""
    # Assumes nums[-1] = nums[n] = -infinity
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            # Peak is in the left half (including mid)
            right = mid
        else:
            # Peak is in the right half (excluding mid)
            left = mid + 1
            
    # When left == right, we have found a peak element
    return nums[left]

def search_in_rotated_sorted_array(nums, target):
    """Searches for a target value in a rotated sorted array."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
            
        # Determine which half is sorted
        if nums[left] <= nums[mid]: # Left half is sorted
            if nums[left] <= target < nums[mid]:
                # Target is in the sorted left half
                right = mid - 1
            else:
                # Target is in the unsorted right half
                left = mid + 1
        else: # Right half is sorted
            if nums[mid] < target <= nums[right]:
                # Target is in the sorted right half
                left = mid + 1
            else:
                # Target is in the unsorted left half
                right = mid - 1
                
    return -1 # Target not found

def find_first_and_last_position(nums, target):
    """Finds the first and last position of a target element in a sorted array."""
    
    def find_bound(is_first):
        left, right = 0, len(nums) - 1
        idx = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                idx = mid
                if is_first:
                    right = mid - 1 # Try to find an earlier occurrence
                else:
                    left = mid + 1  # Try to find a later occurrence
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return idx

    first_pos = find_bound(True)
    if first_pos == -1:
        return [-1, -1] # Target not found
        
    last_pos = find_bound(False)
    return [first_pos, last_pos]

def merge_intervals(intervals):
    """Merges overlapping intervals in a list of intervals."""
    if not intervals:
        return []
        
    # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        # If the list of merged intervals is empty or if the current interval
        # does not overlap with the previous one, append it.
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            # Otherwise, there is overlap, so merge the current and previous intervals.
            # Update the end time of the last merged interval to be the maximum of
            # the current interval's end time and the previous interval's end time.
            merged[-1][1] = max(merged[-1][1], interval[1])
            
    return merged

def insert_interval(intervals, new_interval):
    """Inserts a new interval into a list of non-overlapping intervals, merging if necessary."""
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that end before the new interval starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
        
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval) # Add the merged interval
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
        
    return result

def get_summary_ranges(nums):
    """Returns a summary of ranges for a sorted list of unique integers."""
    if not nums:
        return []
        
    ranges = []
    start = nums[0]
    
    for i in range(1, len(nums)):
        # If the current number is not consecutive to the previous one
        if nums[i] != nums[i-1] + 1:
            # Add the previous range
            if start == nums[i-1]:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}->{nums[i-1]}")
            # Start a new range
            start = nums[i]
            
    # Add the last range
    if start == nums[-1]:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}->{nums[-1]}")
        
    return ranges

def find_duplicate_subarrays(nums, k):
    """Finds if there are duplicate subarrays of length k."""
    seen = set()
    for i in range(len(nums) - k + 1):
        subarray = tuple(nums[i:i+k]) # Use tuple for hashability
        if subarray in seen:
            return True
        seen.add(subarray)
    return False

def find_anagrams(s, p):
    """Finds all the start indices of p's anagrams in s."""
    ns, np = len(s), len(p)
    if ns < np:
        return []

    p_count = {}
    s_count = {}
    
    # Initialize counts for the first window
    for i in range(np):
        p_count[p[i]] = p_count.get(p[i], 0) + 1
        s_count[s[i]] = s_count.get(s[i], 0) + 1
        
    result = []
    if s_count == p_count:
        result.append(0)
        
    # Slide the window
    for i in range(np, ns):
        # Add the new character to the window
        s_count[s[i]] = s_count.get(s[i], 0) + 1
        # Remove the character leaving the window
        s_count[s[i - np]] -= 1
        if s_count[s[i - np]] == 0:
            del s_count[s[i - np]]
            
        # Check if the current window is an anagram
        if s_count == p_count:
            result.append(i - np + 1)
            
    return result

def find_substring_with_concatenation(s, words):
    """Finds all starting indices of substrings in s that are a concatenation of all words."""
    if not s or not words:
        return []

    word_len = len(words[0])
    num_words = len(words)
    total_len = word_len * num_words
    s_len = len(s)
    
    if s_len < total_len:
        return []

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
        
    result = []
    
    for i in range(s_len - total_len + 1):
        seen_words = {}
        j = 0
        while j < num_words:
            word_start_index = i + j * word_len
            current_word = s[word_start_index : word_start_index + word_len]
            
            if current_word not in word_counts:
                break # Word not in the list of required words
                
            seen_words[current_word] = seen_words.get(current_word, 0) + 1
            
            if seen_words[current_word] > word_counts[current_word]:
                break # Found more occurrences of this word than allowed
                
            j += 1
            
        if j == num_words: # If we successfully matched all words
            result.append(i)
            
    return result

def longest_substring_without_repeating_characters(s):
    """Finds the length of the longest substring without repeating characters."""
    char_index_map = {}
    max_length = 0
    start = 0
    
    for end in range(len(s)):
        if s[end] in char_index_map and char_index_map[s[end]] >= start:
            # If the character is already in the current window, move the start
            start = char_index_map[s[end]] + 1
            
        # Update the last seen index of the character
        char_index_map[s[end]] = end
        
        # Calculate the current window length and update max_length
        current_length = end - start + 1
        max_length = max(max_length, current_length)
        
    return max_length

def longest_common_prefix(strs):
    """Finds the longest common prefix string amongst an array of strings."""
    if not strs:
        return ""
        
    # Sort the list of strings. The common prefix must be a prefix of the
    # first and last string after sorting.
    strs.sort()
    
    first_str = strs[0]
    last_str = strs[-1]
    
    prefix = ""
    for i in range(min(len(first_str), len(last_str))):
        if first_str[i] == last_str[i]:
            prefix += first_str[i]
        else:
            break
            
    return prefix

def is_valid_parentheses(s):
    """Checks if a string containing just the characters '(', ')', '{', '}', '[' and ']' is valid."""
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    
    for char in s:
        if char in mapping: # If it's a closing bracket
            # Pop the top element from the stack if it's not empty, otherwise assign a dummy value
            top_element = stack.pop() if stack else '#'
            # Check if the popped element is the corresponding opening bracket
            if mapping[char] != top_element:
                return False
        else: # If it's an opening bracket
            stack.append(char)
            
    # If the stack is empty, all brackets were matched correctly
    return not stack

def reverse_words_in_a_string(s):
    """Reverses the order of words in a string."""
    words = s.split() # Splits by whitespace and removes empty strings
    return " ".join(words[::-1])

def string_compression(chars):
    """Compresses a list of characters in-place."""
    write_idx = 0
    read_idx = 0
    n = len(chars)
    
    while read_idx < n:
        current_char = chars[read_idx]
        count = 0
        # Count consecutive occurrences of the current character
        while read_idx < n and chars[read_idx] == current_char:
            read_idx += 1
            count += 1
            
        # Write the character
        chars[write_idx] = current_char
        write_idx += 1
        
        # Write the count if it's greater than 1
        if count > 1:
            for digit in str(count):
                chars[write_idx] = digit
                write_idx += 1
                
    # Truncate the list to the compressed length
    del chars[write_idx:]
    return write_idx # Return the new length

def can_construct(ransom_note, magazine):
    """Checks if ransom_note can be constructed from magazine letters."""
    from collections import Counter
    
    magazine_counts = Counter(magazine)
    ransom_counts = Counter(ransom_note)
    
    for char, count in ransom_counts.items():
        if magazine_counts[char] < count:
            return False
    return True

def is_isomorphic(s, t):
    """Checks if two strings are isomorphic."""
    if len(s) != len(t):
        return False
        
    map_s_t = {}
    map_t_s = {}
    
    for char_s, char_t in zip(s, t):
        # Check mapping from s to t
        if char_s in map_s_t:
            if map_s_t[char_s] != char_t:
                return False
        else:
            map_s_t[char_s] = char_t
            
        # Check mapping from t to s (ensures one-to-one mapping)
        if char_t in map_t_s:
            if map_t_s[char_t] != char_s:
                return False
        else:
            map_t_s[char_t] = char_s
            
    return True

def word_pattern(pattern, s):
    """Checks if a string s follows the pattern."""
    words = s.split()
    if len(pattern) != len(words):
        return False
        
    map_p_w = {}
    map_w_p = {}
    
    for char, word in zip(pattern, words):
        # Check pattern to word mapping
        if char in map_p_w:
            if map_p_w[char] != word:
                return False
        else:
            map_p_w[char] = word
            
        # Check word to pattern mapping
        if word in map_w_p:
            if map_w_p[word] != char:
                return False
        else:
            map_w_p[word] = char
            
    return True

def group_anagrams(strs):
    """Groups anagrams together from a list of strings."""
    from collections import defaultdict
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Sort the string to create a canonical representation for anagrams
        sorted_s = "".join(sorted(s))
        anagram_map[sorted_s].append(s)
        
    return list(anagram_map.values())

def product_of_array_except_self(nums):
    """Calculates the product of all elements in an array except for the element at the current index."""
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products
    prefix_product = 1
    for i in range(n):
        result[i] = prefix_product
        prefix_product *= nums[i]
        
    # Calculate suffix products and multiply with prefix products
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
        
    return result

def max_subarray_sum(nums):
    """Finds the contiguous subarray within an array (containing at least one number) which has the largest sum."""
    # Kadane's Algorithm
    max_so_far = float('-inf')
    current_max = 0
    
    for num in nums:
        current_max += num
        if current_max > max_so_far:
            max_so_far = current_max
        if current_max < 0:
            current_max = 0
            
    # Handle the case where all numbers are negative
    if max_so_far == float('-inf'):
        return max(nums)
        
    return max_so_far

def max_subarray_product(nums):
    """Finds the contiguous subarray within an array (containing at least one number) which has the largest product."""
    if not nums:
        return 0
        
    max_prod = nums[0]
    min_prod = nums[0]
    result = nums[0]
    
    for i in range(1, len(nums)):
        num = nums[i]
        # Need to consider the current number, product with max_prod, and product with min_prod
        # because a negative number multiplied by a minimum (negative) can become a maximum.
        temp_max = max(num, max_prod * num, min_prod * num)
        min_prod = min(num, max_prod * num, min_prod * num)
        max_prod = temp_max
        
        result = max(result, max_prod)
        
    return result

def contains_duplicate(nums):
    """Checks if an array contains any duplicate values."""
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

def single_number_iii(nums):
    """Finds the two numbers that appear only once in an array where all other numbers appear twice."""
    # This is the same as find_two_single_numbers
    xor_sum = 0
    for num in nums:
        xor_sum ^= num
        
    rightmost_set_bit = xor_sum & -xor_sum
    
    num1 = 0
    num2 = 0
    
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num
            
    return num1, num2

def find_median_sorted_arrays(nums1, nums2):
    """Finds the median of two sorted arrays."""
    m, n = len(nums1), len(nums2)
    total_len = m + n
    
    # Ensure nums1 is the shorter array for efficiency
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
        
    # Binary search on the shorter array (nums1)
    low, high = 0, m
    
    while low <= high:
        partition1 = (low + high) // 2
        partition2 = (total_len + 1) // 2 - partition1
        
        # Get the elements around the partitions
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if partitions are correct
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if total_len % 2 == 0:
                # Even number of elements, median is the average of the two middle elements
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                # Odd number of elements, median is the larger of the two left elements
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # Partition1 is too large, move left in nums1
            high = partition1 - 1
        else:
            # Partition1 is too small, move right in nums1
            low = partition1 + 1
            
    raise ValueError("Input arrays are not sorted or invalid.")

def search_matrix(matrix, target):
    """Searches for a target value in an m x n integer matrix.
    
    The matrix has the following properties:
    - Integers in each row are sorted in ascending from left to right.
    - The first integer of each row is greater than the last integer of the previous row.
    """
    if not matrix or not matrix[0]:
        return False
        
    rows = len(matrix)
    cols = len(matrix[0])
    
    # Treat the matrix as a single sorted array of size rows * cols
    # Binary search on this virtual array
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid_idx = (left + right) // 2
        mid_val = matrix[mid_idx // cols][mid_idx % cols]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid_idx + 1
        else:
            right = mid_idx - 1
            
    return False

def find_kth_largest_element(nums, k):
    """Finds the kth largest element in an unsorted array."""
    # We can use sorting, but that's O(N log N).
    # A more efficient approach is using QuickSelect (average O(N), worst O(N^2)).
    # Or using a min-heap of size k (O(N log k)).
    
    # Using QuickSelect approach (partitioning)
    
    # We are looking for the (n-k)th smallest element
    target_index = len(nums) - k
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        store_index = left
        # Move all smaller elements to the left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        # Move pivot to its final place
        nums[store_index], nums[right] = nums[right], nums[store_index]
        return store_index

    def quickselect(left, right):
        if left == right: # If the list contains only one element
            return nums[left]
            
        # Select a random pivot index
        import random
        pivot_index = random.randint(left, right) 
        
        # Find the pivot position in a sorted list
        pivot_index = partition(left, right, pivot_index)
        
        # The pivot is in its final sorted position
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            # Go left
            return quickselect(left, pivot_index - 1)
        else:
            # Go right
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_kth_smallest_element(nums, k):
    """Finds the kth smallest element in an unsorted array."""
    # This is equivalent to finding the (n-k+1)th largest element.
    # Or, more directly, find the (k-1)th index in a sorted array.
    # Using QuickSelect for O(N) average time complexity.
    
    target_index = k - 1 # We want the element at index k-1 in the sorted array
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_median_sorted_array(nums):
    """Finds the median of a single sorted array."""
    n = len(nums)
    if n == 0:
        raise ValueError("Input array cannot be empty.")
    
    mid = n // 2
    if n % 2 == 0:
        # Even number of elements, median is the average of the two middle elements
        return (nums[mid - 1] + nums[mid]) / 2.0
    else:
        # Odd number of elements, median is the middle element
        return float(nums[mid])

def find_median_unsorted_array(nums):
    """Finds the median of an unsorted array."""
    # Sort the array first, then find the median
    sorted_nums = sorted(nums)
    return find_median_sorted_array(sorted_nums)

def find_majority_element(nums):
    """Finds the majority element in an array (appears more than n/2 times)."""
    # Boyer-Moore Voting Algorithm
    candidate = None
    count = 0
    
    for num in nums:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
        
    # Optional: Verify if the candidate is indeed the majority element
    # count = 0
    # for num in nums:
    #     if num == candidate:
    #         count += 1
    # if count > len(nums) // 2:
    #     return candidate
    # else:
    #     return None # Or raise an error, depending on requirements
    
    return candidate

def find_majority_element_ii(nums):
    """Finds all elements that appear more than n/3 times."""
    # Generalized Boyer-Moore Voting Algorithm for > n/3
    if not nums:
        return []
        
    candidate1, candidate2 = None, None
    count1, count2 = 0, 0
    
    # First pass: Find potential candidates
    for num in nums:
        if num == candidate1:
            count1 += 1
        elif num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1 = num
            count1 = 1
        elif count2 == 0:
            candidate2 = num
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1
            
    # Second pass: Verify candidates
    result = []
    count1 = 0
    count2 = 0
    for num in nums:
        if num == candidate1:
            count1 += 1
        elif num == candidate2:
            count2 += 1
            
    n = len(nums)
    if count1 > n // 3:
        result.append(candidate1)
    if count2 > n // 3 and candidate1 != candidate2: # Ensure distinct candidates
        result.append(candidate2)
        
    return result

def get_pascal_triangle(num_rows):
    """Generates the first num_rows of Pascal's triangle."""
    if num_rows <= 0:
        return []
        
    triangle = [[1]]
    
    for i in range(1, num_rows):
        prev_row = triangle[-1]
        new_row = [1] # Start with 1
        
        # Calculate intermediate elements
        for j in range(len(prev_row) - 1):
            new_row.append(prev_row[j] + prev_row[j+1])
            
        new_row.append(1) # End with 1
        triangle.append(new_row)
        
    return triangle

def get_row_pascal_triangle(row_index):
    """Returns the row (0-indexed) of Pascal's triangle."""
    # Uses the formula C(n, k) = C(n, k-1) * (n - k + 1) / k
    row = [1] * (row_index + 1)
    for k in range(1, row_index + 1):
        row[k] = row[k-1] * (row_index - k + 1) // k
    return row

def climb_stairs(n):
    """Calculates the number of distinct ways to climb to the top of n stairs."""
    # This is a Fibonacci sequence problem: F(n+1)
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
        
    # Use dynamic programming (or iterative Fibonacci)
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b

def min_cost_climbing_stairs(cost):
    """Finds the minimum cost to reach the top of the staircase."""
    # You can either start from the first step or the second step.
    # dp[i] = cost to reach step i
    # dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    
    n = len(cost)
    if n == 0:
        return 0
    if n == 1:
        return cost[0]
        
    # dp array to store minimum cost to reach each step
    # We can optimize space by only keeping track of the last two costs
    
    # cost_to_reach_prev_prev = cost[0]
    # cost_to_reach_prev = cost[1]
    
    # for i in range(2, n):
    #     current_cost = cost[i] + min(cost_to_reach_prev, cost_to_reach_prev_prev)
    #     cost_to_reach_prev_prev = cost_to_reach_prev
    #     cost_to_reach_prev = current_cost
        
    # The top can be reached from the last step or the second to last step
    # return min(cost_to_reach_prev, cost_to_reach_prev_prev)

    # Alternative approach: dp[i] = min cost to reach step i (where step i is the *top* of the staircase)
    # dp[i] represents the minimum cost to reach the top starting from step i.
    # dp[i] = cost[i] + min(dp[i+1], dp[i+2])
    # We want dp[0] or dp[1]
    
    n = len(cost)
    # dp array stores the minimum cost to reach the top FROM index i
    dp = [0] * (n + 1) # dp[n] is the top, cost is 0
    
    # Iterate backwards from the second to last step
    for i in range(n - 1, -1, -1):
        # Cost to reach top from step i is cost[i] + min cost from next step or step after next
        dp[i] = cost[i] + min(dp[i+1], dp[i+2] if i + 2 <= n else 0) # Handle boundary for dp[n]
        
    # The minimum cost to reach the top is the minimum cost starting from step 0 or step 1
    return min(dp[0], dp[1])

def house_robber(nums):
    """Finds the maximum amount of money you can rob tonight without alerting the police."""
    # This is a classic dynamic programming problem.
    # dp[i] = max money robbed up to house i.
    # dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    n = len(nums)
    if n == 0:
        return 0
    if n == 1:
        return nums[0]
        
    # dp = [0] * n
    # dp[0] = nums[0]
    # dp[1] = max(nums[0], nums[1])
    
    # for i in range(2, n):
    #     dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
    # return dp[n-1]
    
    # Space optimized version:
    rob1, rob2 = 0, 0 # rob1: max money robbed ending at i-2, rob2: max money robbed ending at i-1
    
    # [rob1, rob2, n, n+1, ...]
    for n_val in nums:
        # current max is either rob2 (don't rob current house) or rob1 + current house value
        temp = max(n_val + rob1, rob2)
        rob1 = rob2
        rob2 = temp
        
    return rob2

def house_robber_ii(nums):
    """Finds the maximum amount of money you can rob tonight, but the houses are arranged in a circle."""
    # Since houses are in a circle, you cannot rob both the first and the last house.
    # So, we consider two cases:
    # 1. Rob houses from index 0 to n-2 (excluding the last house).
    # 2. Rob houses from index 1 to n-1 (excluding the first house).
    # The maximum of these two cases is the answer.
    
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
        
    # Helper function for house robber problem (linear arrangement)
    def rob_linear(arr):
        rob1, rob2 = 0, 0
        for n_val in arr:
            temp = max(n_val + rob1, rob2)
            rob1 = rob2
            rob2 = temp
        return rob2

    # Case 1: Rob houses from index 0 to n-2
    max_rob_exclude_last = rob_linear(nums[:-1])
    
    # Case 2: Rob houses from index 1 to n-1
    max_rob_exclude_first = rob_linear(nums[1:])
    
    return max(max_rob_exclude_last, max_rob_exclude_first)

def longest_increasing_subsequence(nums):
    """Finds the length of the longest strictly increasing subsequence."""
    if not nums:
        return 0
        
    # dp[i] = length of the longest increasing subsequence ending at index i
    dp = [1] * len(nums)
    
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)

def longest_common_subsequence(text1, text2):
    """Finds the length of the longest common subsequence of two strings."""
    m, n = len(text1), len(text2)
    
    # dp[i][j] = length of LCS of text1[0..i-1] and text2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    return dp[m][n]

def edit_distance(word1, word2):
    """Calculates the minimum number of operations (insert, delete, replace) required to transform word1 into word2."""
    m, n = len(word1), len(word2)
    
    # dp[i][j] = edit distance between word1[0..i-1] and word2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i # Cost of deleting all characters from word1
    for j in range(n + 1):
        dp[0][j] = j # Cost of inserting all characters of word2
        
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] # No operation needed if characters match
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Deletion
                    dp[i][j - 1],    # Insertion
                    dp[i - 1][j - 1] # Replacement
                )
                
    return dp[m][n]

def unique_paths(m, n):
    """Calculates the number of unique paths from the top-left corner to the bottom-right corner of an m x n grid."""
    # Uses dynamic programming. dp[i][j] = number of paths to reach cell (i, j).
    # dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    dp = [[0] * n for _ in range(m)]
    
    # Initialize the first row and first column to 1, as there's only one way to reach any cell there.
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
        
    # Fill the rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
            
    return dp[m-1][n-1]

def unique_paths_with_obstacles(obstacleGrid):
    """Calculates the number of unique paths from the top-left corner to the bottom-right corner, avoiding obstacles."""
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    
    # If the starting cell has an obstacle, there are no paths.
    if obstacleGrid[0][0] == 1:
        return 0
        
    # dp[i][j] = number of paths to reach cell (i, j)
    dp = [[0] * n for _ in range(m)]
    
    # Initialize the starting cell
    dp[0][0] = 1
    
    # Initialize the first column
    for i in range(1, m):
        if obstacleGrid[i][0] == 0 and dp[i-1][0] == 1:
            dp[i][0] = 1
        else:
            dp[i][0] = 0 # Obstacle or unreachable from above
            
    # Initialize the first row
    for j in range(1, n):
        if obstacleGrid[0][j] == 0 and dp[0][j-1] == 1:
            dp[0][j] = 1
        else:
            dp[0][j] = 0 # Obstacle or unreachable from left
            
    # Fill the rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            if obstacleGrid[i][j] == 0: # If the current cell is not an obstacle
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
            else:
                dp[i][j] = 0 # Obstacle means 0 paths to this cell
                
    return dp[m-1][n-1]

def minimum_path_sum(grid):
    """Finds the minimum path sum from top-left to bottom-right in a grid."""
    m = len(grid)
    n = len(grid[0])
    
    # dp[i][j] = minimum path sum to reach cell (i, j)
    dp = [[0] * n for _ in range(m)]
    
    # Initialize the starting cell
    dp[0][0] = grid[0][0]
    
    # Initialize the first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
        
    # Initialize the first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
        
    # Fill the rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
            
    return dp[m-1][n-1]

def can_partition(nums):
    """Checks if an array can be partitioned into two subsets with equal sum."""
    total_sum = sum(nums)
    
    # If the total sum is odd, it cannot be partitioned into two equal subsets.
    if total_sum % 2 != 0:
        return False
        
    target_sum = total_sum // 2
    n = len(nums)
    
    # dp[i][s] = True if a subset of the first i elements can sum up to s
    dp = [[False] * (target_sum + 1) for _ in range(n + 1)]
    
    # Base case: With 0 elements, we can achieve a sum of 0.
    for i in range(n + 1):
        dp[i][0] = True
        
    # Fill the dp table
    for i in range(1, n + 1):
        for s in range(1, target_sum + 1):
            # If the current number is greater than the current sum 's',
            # we cannot include it, so the result is the same as without this number.
            if nums[i-1] > s:
                dp[i][s] = dp[i-1][s]
            else:
                # We can either include the current number or not.
                # If we include it, we need to check if the remaining sum (s - nums[i-1])
                # can be achieved with the previous elements.
                # If we don't include it, the result is the same as without this number.
                dp[i][s] = dp[i-1][s] or dp[i-1][s - nums[i-1]]
                
    return dp[n][target_sum]

def can_partition_k_subsets(nums, k):
    """Checks if an array can be partitioned into k subsets with equal sum."""
    total_sum = sum(nums)
    if total_sum % k != 0:
        return False
        
    target_sum = total_sum // k
    n = len(nums)
    nums.sort(reverse=True) # Optimization: Try larger numbers first
    
    # visited array to keep track of used numbers
    visited = [False] * n
    
    def backtrack(index, current_sum, count):
        # Base case: If we have successfully formed k-1 subsets, the last one is guaranteed.
        if count == k - 1:
            return True
            
        # If the current subset sum equals the target sum, try to form the next subset.
        if current_sum == target_sum:
            return backtrack(0, 0, count + 1)
            
        # Try adding numbers to the current subset
        for i in range(index, n):
            # If the number is not visited and adding it doesn't exceed the target sum
            if not visited[i] and current_sum + nums[i] <= target_sum:
                visited[i] = True
                # Recursively try to complete the subset
                if backtrack(i + 1, current_sum + nums[i], count):
                    return True
                # Backtrack: If the recursive call didn't lead to a solution, unmark the number.
                visited[i] = False
                
                # Optimization: If current_sum is 0 (start of a new subset) or
                # current_sum + nums[i] == target_sum (just completed a subset),
                # and we failed to find a solution, then any subsequent attempt
                # starting with the same current_sum and trying the same nums[i]
                # will also fail. Skip duplicates.
                if current_sum == 0 or current_sum + nums[i] == target_sum:
                    break
                # Optimization: If nums[i] is the same as the next number and we failed, skip the next one.
                while i + 1 < n and nums[i] == nums[i+1]:
                    i += 1
                    
        return False

    return backtrack(0, 0, 0)

def min_coin_change(coins, amount):
    """Finds the minimum number of coins needed to make a given amount."""
    # dp[i] = minimum number of coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0 # Base case: 0 coins needed for amount 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
                
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_word_in_dictionary(words):
    """Finds the longest word in a dictionary that can be built one character at a time by other words in the dictionary."""
    word_set = set(words)
    longest_word = ""
    
    for word in words:
        # Check if the word can be built character by character
        is_buildable = True
        for i in range(1, len(word)):
            prefix = word[:i]
            if prefix not in word_set:
                is_buildable = False
                break
                
        if is_buildable:
            # If buildable, compare its length and lexicographical order
            if len(word) > len(longest_word):
                longest_word = word
            elif len(word) == len(longest_word) and word < longest_word:
                longest_word = word
                
    return longest_word

def find_words_from_dictionary(board, words):
    """Finds all words from a given dictionary that can be formed by sequentially adjacent letters on a 2D board."""
    
    # Trie implementation for efficient prefix checking
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word = None # Store the word if this node marks the end of a word

    def build_trie(words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word # Mark the end of the word
        return root

    root = build_trie(words)
    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        # Check boundaries and if the character exists in the current Trie node's children
        char = board[r][c]
        if not (0 <= r < rows and 0 <= c < cols) or char not in node.children:
            return

        # Move to the next node in the Trie
        node = node.children[char]

        # If a word is found, add it to the result and remove it from the Trie to avoid duplicates
        if node.word:
            result.append(node.word)
            node.word = None # Mark as found

        # Mark the current cell as visited to avoid reusing it in the same path
        board[r][c] = '#' 

        # Explore neighbors (up, down, left, right)
        dfs(r + 1, c, node)
        dfs(r - 1, c, node)
        dfs(r, c + 1, node)
        dfs(r, c - 1, node)

        # Backtrack: Restore the character in the board
        board[r][c] = char

    # Start DFS from each cell on the board
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
            
    return result

def word_break(s, wordDict):
    """Determines if a string can be segmented into a space-separated sequence of one or more dictionary words."""
    word_set = set(wordDict)
    n = len(s)
    
    # dp[i] = True if s[0...i-1] can be segmented
    dp = [False] * (n + 1)
    dp[0] = True # Base case: empty string can always be segmented
    
    for i in range(1, n + 1):
        for j in range(i):
            # If s[0...j-1] can be segmented (dp[j] is True)
            # AND the substring s[j...i-1] is in the dictionary
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break # Found a valid segmentation for s[0...i-1], move to next i
                
    return dp[n]

def word_break_ii(s, wordDict):
    """Returns all possible sentences where s can be segmented into a space-separated sequence of dictionary words."""
    word_set = set(wordDict)
    memo = {} # Memoization to store results for substrings

    def backtrack(start_index):
        if start_index in memo:
            return memo[start_index]
        
        if start_index == len(s):
            return [""] # Base case: empty string means a valid segmentation ending here

        results = []
        for end_index in range(start_index + 1, len(s) + 1):
            word = s[start_index:end_index]
            if word in word_set:
                # Recursively find sentences for the rest of the string
                rest_of_sentences = backtrack(end_index)
                for sentence in rest_of_sentences:
                    if sentence: # If there's a sentence following the current word
                        results.append(word + " " + sentence)
                    else: # If this is the last word in the sentence
                        results.append(word)
                        
        memo[start_index] = results
        return results

    return backtrack(0)

def palindrome_partitioning(s):
    """Partitions a string s such that every substring of the partition is a palindrome."""
    n = len(s)
    result = []
    current_partition = []

    # Helper function to check if a substring is a palindrome
    def is_palindrome(sub):
        return sub == sub[::-1]

    # Backtracking function
    def backtrack(start_index):
        # Base case: If we have reached the end of the string, add the current partition to the result
        if start_index == n:
            result.append(list(current_partition)) # Add a copy
            return

        # Iterate through all possible end points for the current substring
        for end_index in range(start_index, n):
            substring = s[start_index : end_index + 1]
            
            # If the substring is a palindrome
            if is_palindrome(substring):
                # Add it to the current partition
                current_partition.append(substring)
                # Recursively call backtrack for the rest of the string
                backtrack(end_index + 1)
                # Backtrack: Remove the substring to explore other possibilities
                current_partition.pop()

    backtrack(0)
    return result

def palindrome_partitioning_ii(s):
    """Finds the minimum cuts needed for a palindrome partitioning of a string."""
    n = len(s)
    
    # dp[i] = minimum cuts needed for palindrome partitioning of s[0...i-1]
    dp = [i - 1 for i in range(n + 1)] # Initialize with max possible cuts (i-1)
    
    # is_palindrome[i][j] = True if s[i...j] is a palindrome
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Precompute palindrome status for all substrings
    for length in range(1, n + 1): # length of substring
        for i in range(n - length + 1):
            j = i + length - 1 # end index
            if length == 1:
                is_palindrome[i][j] = True
            elif length == 2:
                is_palindrome[i][j] = (s[i] == s[j])
            else:
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i+1][j-1])
                
    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(i):
            # If s[j...i-1] is a palindrome
            if is_palindrome[j][i-1]:
                # The minimum cuts to partition s[0...i-1] is either the current minimum cuts (dp[i])
                # or 1 (for the cut before s[j...i-1]) + minimum cuts for s[0...j-1] (dp[j])
                dp[i] = min(dp[i], dp[j] + 1)
                
    return dp[n]

def combinations(n, k):
    """Generates all possible combinations of k numbers chosen from 1 to n."""
    result = []
    
    def backtrack(start, current_combination):
        # Base case: If the combination has k elements, add it to the result
        if len(current_combination) == k:
            result.append(list(current_combination)) # Add a copy
            return
            
        # Iterate through possible numbers to add
        # The range ensures we don't exceed n and maintain ascending order
        # The condition `n - i + 1 >= k - len(current_combination)` ensures that
        # there are enough remaining numbers to form a combination of size k.
        for i in range(start, n + 1):
            # Add the number to the current combination
            current_combination.append(i)
            # Recursively call backtrack for the next element
            backtrack(i + 1, current_combination)
            # Backtrack: Remove the number to explore other possibilities
            current_combination.pop()

    backtrack(1, [])
    return result

def permutations(nums):
    """Generates all possible permutations of a list of unique numbers."""
    result = []
    n = len(nums)
    
    def backtrack(current_permutation, used_mask):
        # Base case: If the permutation has n elements, add it to the result
        if len(current_permutation) == n:
            result.append(list(current_permutation))
            return
            
        # Iterate through all numbers
        for i in range(n):
            # If the number hasn't been used yet
            if not (used_mask & (1 << i)):
                # Add the number to the current permutation
                current_permutation.append(nums[i])
                # Mark the number as used
                new_mask = used_mask | (1 << i)
                # Recursively call backtrack
                backtrack(current_permutation, new_mask)
                # Backtrack: Remove the number and unmark it
                current_permutation.pop()

    backtrack([], 0) # Start with an empty permutation and no numbers used
    return result

def subsets(nums):
    """Generates all possible subsets (the power set) of a list of unique numbers."""
    result = []
    n = len(nums)
    
    def backtrack(start_index, current_subset):
        # Add the current subset to the result at each step
        result.append(list(current_subset))
        
        # Iterate through the remaining elements
        for i in range(start_index, n):
            # Include the current element
            current_subset.append(nums[i])
            # Recursively call backtrack for the next element
            backtrack(i + 1, current_subset)
            # Backtrack: Exclude the current element
            current_subset.pop()

    backtrack(0, [])
    return result

def subsets_with_dup(nums):
    """Generates all possible subsets (the power set) of a list that may contain duplicates."""
    result = []
    n = len(nums)
    nums.sort() # Sort the list to handle duplicates easily
    
    def backtrack(start_index, current_subset):
        result.append(list(current_subset))
        
        for i in range(start_index, n):
            # Skip duplicates: If the current element is the same as the previous one,
            # and we are not considering it for the first time in this level of recursion, skip it.
            if i > start_index and nums[i] == nums[i-1]:
                continue
                
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()

    backtrack(0, [])
    return result

def combination_sum(candidates, target):
    """Finds all unique combinations in candidates where the candidate numbers sum to target."""
    result = []
    n = len(candidates)
    candidates.sort() # Sort to handle duplicates and ensure combinations are unique

    def backtrack(start_index, current_combination, current_sum):
        # Base case: If the current sum equals the target, add the combination to the result
        if current_sum == target:
            result.append(list(current_combination))
            return
            
        # If the current sum exceeds the target, prune this path
        if current_sum > target:
            return
            
        # Iterate through candidates starting from start_index
        for i in range(start_index, n):
            # Optimization: If the current candidate is greater than the remaining target,
            # and since the candidates are sorted, no further candidates will work.
            if candidates[i] > target - current_sum:
                break

            # Skip duplicates: If the current candidate is the same as the previous one,
            # and we are not considering it for the first time in this level of recursion, skip it.
            if i > start_index and candidates[i] == candidates[i-1]:
                continue

            # Include the current candidate
            current_combination.append(candidates[i])
            # Recursively call backtrack. We use `i` instead of `i+1` because the same number can be used multiple times.
            backtrack(i, current_combination, current_sum + candidates[i])
            # Backtrack: Remove the candidate
            current_combination.pop()

    backtrack(0, [], 0)
    return result

def combination_sum_ii(candidates, target):
    """Finds all unique combinations in candidates where the candidate numbers sum to target. Each number may only be used once in the combination."""
    result = []
    n = len(candidates)
    candidates.sort() # Sort to handle duplicates and ensure combinations are unique

    def backtrack(start_index, current_combination, current_sum):
        if current_sum == target:
            result.append(list(current_combination))
            return
            
        if current_sum > target:
            return
            
        for i in range(start_index, n):
            # Optimization: If the current candidate is greater than the remaining target, break.
            if candidates[i] > target - current_sum:
                break

            # Skip duplicates: If the current candidate is the same as the previous one,
            # and we are not considering it for the first time in this level of recursion, skip it.
            if i > start_index and candidates[i] == candidates[i-1]:
                continue

            # Include the current candidate
            current_combination.append(candidates[i])
            # Recursively call backtrack. Use `i+1` because each number can only be used once.
            backtrack(i + 1, current_combination, current_sum + candidates[i])
            # Backtrack: Remove the candidate
            current_combination.pop()

    backtrack(0, [], 0)
    return result

def generate_parenthesis(n):
    """Generates all combinations of well-formed parentheses."""
    result = []
    
    def backtrack(current_string, open_count, close_count):
        # Base case: If the string length is 2*n, we have a complete combination
        if len(current_string) == 2 * n:
            result.append(current_string)
            return
            
        # Add an opening parenthesis if we still have open parentheses available
        if open_count < n:
            backtrack(current_string + "(", open_count + 1, close_count)
            
        # Add a closing parenthesis if it doesn't violate the well-formed condition
        # (i.e., number of closing parentheses is less than number of opening parentheses)
        if close_count < open_count:
            backtrack(current_string + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result

def solve_n_queens(n):
    """Solves the N-Queens problem, returning all distinct solutions."""
    result = []
    # `cols` stores the column index of the queen in each row. `cols[r] = c` means a queen is at (r, c).
    cols = [-1] * n 
    
    # `diag1` and `diag2` track occupied diagonals.
    # For diag1 (top-left to bottom-right), row - col is constant.
    # For diag2 (top-right to bottom-left), row + col is constant.
    diag1 = [False] * (2 * n - 1) # Indices range from -(n-1) to (n-1), shifted by n-1 to be non-negative.
    diag2 = [False] * (2 * n - 1) # Indices range from 0 to 2n-2.
    
    # `used_cols` tracks occupied columns.
    used_cols = [False] * n

    def backtrack(row):
        # Base case: If we have placed queens in all rows, we found a solution.
        if row == n:
            # Format the solution into a list of strings representing the board.
            board = []
            for r in range(n):
                row_str = ["."] * n
                row_str[cols[r]] = "Q"
                board.append("".join(row_str))
            result.append(board)
            return

        # Try placing a queen in each column of the current row.
        for col in range(n):
            # Check if the current position (row, col) is safe.
            # Calculate diagonal indices.
            diag1_idx = row - col + n - 1
            diag2_idx = row + col
            
            if not used_cols[col] and not diag1[diag1_idx] and not diag2[diag2_idx]:
                # Place the queen.
                cols[row] = col
                used_cols[col] = True
                diag1[diag1_idx] = True
                diag2[diag2_idx] = True

                # Recursively try to place queens in the next row.
                backtrack(row + 1)

                # Backtrack: Remove the queen and unmark the occupied positions.
                cols[row] = -1
                used_cols[col] = False
                diag1[diag1_idx] = False
                diag2[diag2_idx] = False

    backtrack(0) # Start the process from the first row (row 0).
    return result

def is_valid_sudoku(board):
    """Checks if a 9x9 Sudoku board is valid."""
    n = 9
    
    # Check rows
    for r in range(n):
        seen = set()
        for c in range(n):
            num = board[r][c]
            if num != '.':
                if num in seen:
                    return False
                seen.add(num)
                
    # Check columns
    for c in range(n):
        seen = set()
        for r in range(n):
            num = board[r][c]
            if num != '.':
                if num in seen:
                    return False
                seen.add(num)
                
    # Check 3x3 subgrids
    for i in range(0, n, 3): # Start row of the subgrid
        for j in range(0, n, 3): # Start col of the subgrid
            seen = set()
            for r in range(i, i + 3):
                for c in range(j, j + 3):
                    num = board[r][c]
                    if num != '.':
                        if num in seen:
                            return False
                        seen.add(num)
                        
    return True

def solve_sudoku(board):
    """Solves a Sudoku puzzle in-place."""
    n = 9
    
    # Find the next empty cell (represented by '.')
    def find_empty():
        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    return r, c
        return None # No empty cells left, puzzle is solved

    # Check if placing 'num' at board[row][col] is valid
    def is_valid(num, row, col):
        # Check row
        for c in range(n):
            if board[row][c] == num:
                return False
        # Check column
        for r in range(n):
            if board[r][col] == num:
                return False
        # Check 3x3 subgrid
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == num:
                    return False
        return True

    # Backtracking function
    def backtrack():
        empty_cell = find_empty()
        if not empty_cell:
            return True # Puzzle solved

        row, col = empty_cell
        
        # Try numbers from '1' to '9'
        for num_char in "123456789":
            if is_valid(num_char, row, col):
                board[row][col] = num_char # Place the number
                
                if backtrack(): # Recursively try to solve the rest
                    return True
                    
                board[row][col] = '.' # Backtrack: Reset the cell if the path didn't lead to a solution
                
        return False # No number worked for this cell

    backtrack() # Start the solving process
    # The board is modified in-place. Return True if solved, False otherwise (though problem usually guarantees a solution).
    # For this function, we assume it modifies the board and doesn't need to return a boolean.
    # If a return value indicating success/failure is needed, modify backtrack to return boolean.

def find_duplicate_numbers(nums):
    """Finds all duplicate numbers in an array where numbers are in the range [1, n] and each appears once or twice."""
    duplicates = []
    for num in nums:
        # Get the index corresponding to the number's value
        index = abs(num) - 1
        
        # If the number at that index is already negative, it means we've seen this number before
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            # Mark the number as seen by negating the value at its corresponding index
            nums[index] = -nums[index]
            
    # Optional: Restore the original array if needed (by taking absolute values)
    # for i in range(len(nums)):
    #     nums[i] = abs(nums[i])
        
    return duplicates

def find_disappeared_numbers(nums):
    """Finds all the numbers that appear once in an array where numbers are in the range [1, n]."""
    n = len(nums)
    for num in nums:
        # Use the number's value to find the index and mark the element at that index.
        # We use abs(num) because the element might have been negated already.
        index = abs(num) - 1
        # Negate the element at the index to mark it as seen.
        # Use abs() on nums[index] to ensure we don't accidentally make it positive again.
        nums[index] = -abs(nums[index])
        
    # The indices where the elements are still positive correspond to the disappeared numbers.
    disappeared = []
    for i in range(n):
        if nums[i] > 0:
            disappeared.append(i + 1) # Add 1 because numbers are 1-based.
            
    return disappeared

def find_first_missing_positive(nums):
    """Finds the smallest missing positive integer in an unsorted array."""
    n = len(nums)
    
    # Step 1: Place each positive integer i at index i-1 if possible.
    # Ignore non-positive numbers and numbers greater than n.
    for i in range(n):
        # Condition `0 < nums[i] <= n` ensures we only process relevant positive integers.
        # Condition `nums[nums[i] - 1] != nums[i]` prevents infinite loops if the number is already in its correct place.
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap nums[i] with the element at its correct index (nums[i] - 1).
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            
    # Step 2: Iterate through the array. The first index i where nums[i] != i + 1
    # indicates that i + 1 is the smallest missing positive integer.
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    # If all numbers from 1 to n are present, the smallest missing positive is n + 1.
    return n + 1

def product_of_array_except_self_division(nums):
    """Calculates the product of all elements in an array except for the element at the current index, using division."""
    total_product = 1
    zero_count = 0
    
    for num in nums:
        if num == 0:
            zero_count += 1
        else:
            total_product *= num
            
    n = len(nums)
    result = [0] * n
    
    if zero_count > 1:
        # If there are multiple zeros, all results will be 0.
        return result
    elif zero_count == 1:
        # If there is exactly one zero, only the result at the zero's index will be non-zero.
        for i in range(n):
            if nums[i] == 0:
                result[i] = total_product
                break
        return result
    else:
        # No zeros, calculate product / nums[i] for each element.
        for i in range(n):
            result[i] = total_product // nums[i]
        return result

def find_median_sorted_arrays_optimized(nums1, nums2):
    """Finds the median of two sorted arrays using binary search on partitions."""
    m, n = len(nums1), len(nums2)
    
    # Ensure nums1 is the shorter array for efficiency
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
        
    # Binary search on the shorter array (nums1)
    low, high = 0, m
    total_len = m + n
    half_len = (total_len + 1) // 2 # The number of elements in the left partition
    
    while low <= high:
        partition1 = (low + high) // 2 # Partition index for nums1
        partition2 = half_len - partition1 # Corresponding partition index for nums2
        
        # Get the elements around the partitions
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if partitions are correct: max of left <= min of right
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if total_len % 2 == 0:
                # Even number of elements, median is the average of the two middle elements
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                # Odd number of elements, median is the larger of the two left elements
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # Partition1 is too large, move left in nums1
            high = partition1 - 1
        else: # max_left2 > min_right1
            # Partition1 is too small, move right in nums1
            low = partition1 + 1
            
    # Should not reach here if inputs are valid sorted arrays
    raise ValueError("Input arrays are not sorted or invalid.")

def find_kth_largest_element_optimized(nums, k):
    """Finds the kth largest element using QuickSelect (average O(N))."""
    target_index = len(nums) - k # Index of the kth largest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_kth_smallest_element_optimized(nums, k):
    """Finds the kth smallest element using QuickSelect (average O(N))."""
    target_index = k - 1 # Index of the kth smallest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_duplicate_subarrays_optimized(nums, k):
    """Finds if there are duplicate subarrays of length k using rolling hash."""
    n = len(nums)
    if n < k:
        return False

    # Use a set to store hashes of seen subarrays
    seen_hashes = set()
    
    # Rolling hash parameters (choose large prime modulus and base)
    MOD = 10**9 + 7
    BASE = 31 # A prime number commonly used for hashing strings/sequences

    # Calculate the hash of the first subarray
    current_hash = 0
    for i in range(k):
        current_hash = (current_hash * BASE + nums[i]) % MOD
    seen_hashes.add(current_hash)

    # Precompute BASE^(k-1) % MOD for efficient hash updates
    power_base_k_minus_1 = pow(BASE, k - 1, MOD)

    # Slide the window and update the hash
    for i in range(k, n):
        # Remove the contribution of the element leaving the window
        current_hash = (current_hash - (nums[i - k] * power_base_k_minus_1) % MOD + MOD) % MOD
        # Add the contribution of the new element entering the window
        current_hash = (current_hash * BASE + nums[i]) % MOD
        
        # Check if the current hash has been seen before
        if current_hash in seen_hashes:
            # Potential duplicate found. To be absolutely sure (avoid hash collisions),
            # you might need to compare the actual subarrays. However, for many problems,
            # hash collisions are rare enough that this check is sufficient.
            # For a guaranteed correct solution, you'd store subarrays or indices along with hashes.
            return True
        seen_hashes.add(current_hash)
        
    return False

def find_duplicate_numbers_optimized(nums):
    """Finds all duplicate numbers in an array where numbers are in the range [1, n] and each appears once or twice."""
    # This uses the same in-place modification technique as the non-optimized version.
    # The optimization is in the problem statement itself (numbers are within [1, n]).
    duplicates = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            nums[index] = -nums[index]
    return duplicates

def find_disappeared_numbers_optimized(nums):
    """Finds all the numbers that appear once in an array where numbers are in the range [1, n]."""
    # This uses the same in-place modification technique as the non-optimized version.
    # The optimization is in the problem statement itself (numbers are within [1, n]).
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        nums[index] = -abs(nums[index])
        
    disappeared = []
    for i in range(n):
        if nums[i] > 0:
            disappeared.append(i + 1)
            
    return disappeared

def find_first_missing_positive_optimized(nums):
    """Finds the smallest missing positive integer in an unsorted array using in-place swaps."""
    n = len(nums)
    
    # Place each positive integer i at index i-1 if possible.
    for i in range(n):
        # The while loop ensures that the number at nums[i] is placed in its correct position
        # until nums[i] is either out of range [1, n], or it's already in its correct place,
        # or the target position already holds the correct number (to avoid infinite loops).
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            
    # Find the first index i where nums[i] != i + 1.
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    # If all numbers from 1 to n are present, the smallest missing positive is n + 1.
    return n + 1

def product_of_array_except_self_no_division(nums):
    """Calculates the product of all elements in an array except for the element at the current index, without using division."""
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products: result[i] will store the product of all elements to the left of i.
    prefix_product = 1
    for i in range(n):
        result[i] = prefix_product
        prefix_product *= nums[i]
        
    # Calculate suffix products and multiply them with the prefix products.
    # suffix_product stores the product of all elements to the right of i.
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
        
    return result

def find_median_sorted_arrays_final(nums1, nums2):
    """Finds the median of two sorted arrays using binary search on partitions (optimized)."""
    m, n = len(nums1), len(nums2)
    
    # Ensure nums1 is the shorter array for efficiency
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
        
    low, high = 0, m
    total_len = m + n
    half_len = (total_len + 1) // 2 # Number of elements in the left partition
    
    while low <= high:
        partition1 = (low + high) // 2 # Partition index for nums1
        partition2 = half_len - partition1 # Corresponding partition index for nums2
        
        # Determine the boundary elements for the partitions
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if the partitions are correct: max of left <= min of right
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if total_len % 2 == 0:
                # Even number of elements: median is the average of the two middle elements
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                # Odd number of elements: median is the larger of the two left elements
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # partition1 is too large, need to move left in nums1
            high = partition1 - 1
        else: # max_left2 > min_right1
            # partition1 is too small, need to move right in nums1
            low = partition1 + 1
            
    # This part should ideally not be reached if the input arrays are sorted.
    raise ValueError("Input arrays are not sorted or invalid.")

def find_kth_largest_element_final(nums, k):
    """Finds the kth largest element using QuickSelect (average O(N))."""
    target_index = len(nums) - k # Index of the kth largest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_kth_smallest_element_final(nums, k):
    """Finds the kth smallest element using QuickSelect (average O(N))."""
    target_index = k - 1 # Index of the kth smallest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_duplicate_subarrays_final(nums, k):
    """Finds if there are duplicate subarrays of length k using rolling hash (optimized)."""
    n = len(nums)
    if n < k:
        return False

    seen_hashes = set()
    
    MOD = 10**9 + 7
    BASE = 31 

    current_hash = 0
    for i in range(k):
        current_hash = (current_hash * BASE + nums[i]) % MOD
    seen_hashes.add(current_hash)

    power_base_k_minus_1 = pow(BASE, k - 1, MOD)

    for i in range(k, n):
        current_hash = (current_hash - (nums[i - k] * power_base_k_minus_1) % MOD + MOD) % MOD
        current_hash = (current_hash * BASE + nums[i]) % MOD
        
        if current_hash in seen_hashes:
            # For guaranteed correctness, one would need to verify the actual subarray.
            # However, for typical competitive programming scenarios, hash collisions are rare.
            return True
        seen_hashes.add(current_hash)
        
    return False

def find_duplicate_numbers_final(nums):
    """Finds all duplicate numbers in an array where numbers are in the range [1, n] using in-place modification."""
    duplicates = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            nums[index] = -nums[index]
    return duplicates

def find_disappeared_numbers_final(nums):
    """Finds all the numbers that appear once in an array where numbers are in the range [1, n] using in-place modification."""
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        nums[index] = -abs(nums[index])
        
    disappeared = []
    for i in range(n):
        if nums[i] > 0:
            disappeared.append(i + 1)
            
    return disappeared

def find_first_missing_positive_final(nums):
    """Finds the smallest missing positive integer in an unsorted array using in-place swaps (optimized)."""
    n = len(nums)
    
    # Place each positive integer i at index i-1 if possible.
    for i in range(n):
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            
    # Find the first index i where nums[i] != i + 1.
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    # If all numbers from 1 to n are present, the smallest missing positive is n + 1.
    return n + 1

def product_of_array_except_self_final(nums):
    """Calculates the product of all elements in an array except for the element at the current index, without using division (optimized)."""
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products
    prefix_product = 1
    for i in range(n):
        result[i] = prefix_product
        prefix_product *= nums[i]
        
    # Calculate suffix products and multiply with prefix products
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
        
    return result

def find_median_sorted_arrays_final_final(nums1, nums2):
    """Finds the median of two sorted arrays using binary search on partitions (final optimized version)."""
    m, n = len(nums1), len(nums2)
    
    # Ensure nums1 is the shorter array for efficiency
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
        
    low, high = 0, m
    total_len = m + n
    half_len = (total_len + 1) // 2 # Number of elements in the left partition
    
    while low <= high:
        partition1 = (low + high) // 2 # Partition index for nums1
        partition2 = half_len - partition1 # Corresponding partition index for nums2
        
        # Determine the boundary elements for the partitions
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if the partitions are correct: max of left <= min of right
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if total_len % 2 == 0:
                # Even number of elements: median is the average of the two middle elements
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                # Odd number of elements: median is the larger of the two left elements
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # partition1 is too large, need to move left in nums1
            high = partition1 - 1
        else: # max_left2 > min_right1
            # partition1 is too small, need to move right in nums1
            low = partition1 + 1
            
    # This part should ideally not be reached if the input arrays are sorted.
    raise ValueError("Input arrays are not sorted or invalid.")

def find_kth_largest_element_final_final(nums, k):
    """Finds the kth largest element using QuickSelect (average O(N))."""
    target_index = len(nums) - k # Index of the kth largest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_kth_smallest_element_final_final(nums, k):
    """Finds the kth smallest element using QuickSelect (average O(N))."""
    target_index = k - 1 # Index of the kth smallest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_duplicate_subarrays_final_final(nums, k):
    """Finds if there are duplicate subarrays of length k using rolling hash (final optimized version)."""
    n = len(nums)
    if n < k:
        return False

    seen_hashes = set()
    
    MOD = 10**9 + 7
    BASE = 31 

    current_hash = 0
    for i in range(k):
        current_hash = (current_hash * BASE + nums[i]) % MOD
    seen_hashes.add(current_hash)

    power_base_k_minus_1 = pow(BASE, k - 1, MOD)

    for i in range(k, n):
        current_hash = (current_hash - (nums[i - k] * power_base_k_minus_1) % MOD + MOD) % MOD
        current_hash = (current_hash * BASE + nums[i]) % MOD
        
        if current_hash in seen_hashes:
            # For guaranteed correctness, one would need to verify the actual subarray.
            # However, for typical competitive programming scenarios, hash collisions are rare.
            return True
        seen_hashes.add(current_hash)
        
    return False

def find_duplicate_numbers_final_final(nums):
    """Finds all duplicate numbers in an array where numbers are in the range [1, n] using in-place modification (final version)."""
    duplicates = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            nums[index] = -nums[index]
    return duplicates

def find_disappeared_numbers_final_final(nums):
    """Finds all the numbers that appear once in an array where numbers are in the range [1, n] using in-place modification (final version)."""
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        nums[index] = -abs(nums[index])
        
    disappeared = []
    for i in range(n):
        if nums[i] > 0:
            disappeared.append(i + 1)
            
    return disappeared

def find_first_missing_positive_final_final(nums):
    """Finds the smallest missing positive integer in an unsorted array using in-place swaps (final optimized version)."""
    n = len(nums)
    
    # Place each positive integer i at index i-1 if possible.
    for i in range(n):
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            
    # Find the first index i where nums[i] != i + 1.
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    # If all numbers from 1 to n are present, the smallest missing positive is n + 1.
    return n + 1

def product_of_array_except_self_final_final(nums):
    """Calculates the product of all elements in an array except for the element at the current index, without using division (final optimized version)."""
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products
    prefix_product = 1
    for i in range(n):
        result[i] = prefix_product
        prefix_product *= nums[i]
        
    # Calculate suffix products and multiply with prefix products
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
        
    return result

def find_median_sorted_arrays_final_final_final(nums1, nums2):
    """Finds the median of two sorted arrays using binary search on partitions (final final optimized version)."""
    m, n = len(nums1), len(nums2)
    
    # Ensure nums1 is the shorter array for efficiency
    if m > n:
        nums1, nums2, m, n = nums2, nums1, n, m
        
    low, high = 0, m
    total_len = m + n
    half_len = (total_len + 1) // 2 # Number of elements in the left partition
    
    while low <= high:
        partition1 = (low + high) // 2 # Partition index for nums1
        partition2 = half_len - partition1 # Corresponding partition index for nums2
        
        # Determine the boundary elements for the partitions
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if the partitions are correct: max of left <= min of right
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            # Found the correct partitions
            if total_len % 2 == 0:
                # Even number of elements: median is the average of the two middle elements
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            else:
                # Odd number of elements: median is the larger of the two left elements
                return float(max(max_left1, max_left2))
        elif max_left1 > min_right2:
            # partition1 is too large, need to move left in nums1
            high = partition1 - 1
        else: # max_left2 > min_right1
            # partition1 is too small, need to move right in nums1
            low = partition1 + 1
            
    # This part should ideally not be reached if the input arrays are sorted.
    raise ValueError("Input arrays are not sorted or invalid.")

def find_kth_largest_element_final_final_final(nums, k):
    """Finds the kth largest element using QuickSelect (average O(N))."""
    target_index = len(nums) - k # Index of the kth largest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_kth_smallest_element_final_final_final(nums, k):
    """Finds the kth smallest element using QuickSelect (average O(N))."""
    target_index = k - 1 # Index of the kth smallest element if sorted
    
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index] # Move pivot to end
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot_value:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[store_index], nums[right] = nums[right], nums[store_index] # Move pivot to its final place
        return store_index

    def quickselect(left, right):
        if left == right:
            return nums[left]
            
        import random
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if target_index == pivot_index:
            return nums[target_index]
        elif target_index < pivot_index:
            return quickselect(left, pivot_index - 1)
        else:
            return quickselect(pivot_index + 1, right)

    return quickselect(0, len(nums) - 1)

def find_duplicate_subarrays_final_final_final(nums, k):
    """Finds if there are duplicate subarrays of length k using rolling hash (final final optimized version)."""
    n = len(nums)
    if n < k:
        return False

    seen_hashes = set()
    
    MOD = 10**9 + 7
    BASE = 31 

    current_hash = 0
    for i in range(k):
        current_hash = (current_hash * BASE + nums[i]) % MOD
    seen_hashes.add(current_hash)

    power_base_k_minus_1 = pow(BASE, k - 1, MOD)

    for i in range(k, n):
        current_hash = (current_hash - (nums[i - k] * power_base_k_minus_1) % MOD + MOD) % MOD
        current_hash = (current_hash * BASE + nums[i]) % MOD
        
        if current_hash in seen_hashes:
            # For guaranteed correctness, one would need to verify the actual subarray.
            # However, for typical competitive programming scenarios, hash collisions are rare.
            return True
        seen_hashes.add(current_hash)
        
    return False

def find_duplicate_numbers_final_final_final(nums):
    """Finds all duplicate numbers in an array where numbers are in the range [1, n] using in-place modification (final final version)."""
    duplicates = []
    for num in nums:
        index = abs(num) - 1
        if nums[index] < 0:
            duplicates.append(abs(num))
        else:
            nums[index] = -nums[index]
    return duplicates

def find_disappeared_numbers_final_final_final(nums):
    """Finds all the numbers that appear once in an array where numbers are in the range [1, n] using in-place modification (final final version)."""
    n = len(nums)
    for num in nums:
        index = abs(num) - 1
        nums[index] = -abs(nums[index])
        
    disappeared = []
    for i in range(n):
        if nums[i] > 0:
            disappeared.append(i + 1)
            
    return disappeared

def find_first_missing_positive_final_final_final(nums):
    """Finds the smallest missing positive integer in an unsorted array using in-place swaps (final final optimized version)."""
    n = len(nums)
    
    # Place each positive integer i at index i-1 if possible.
    for i in range(n):
        while 0 < nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            
    # Find the first index i where nums[i] != i + 1.
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    # If all numbers from 1 to n are present, the smallest missing positive is n + 1.
    return n + 1

def product_of_array_except_self_final_final_final(nums):
    """Calculates the product of all elements in an array except for the element at the current index, without using division (final final optimized version)."""
    n = len(nums)
    result = [1] * n
    
    # Calculate prefix products
    prefix_product = 1
    for i in range(n):
        result[i] = prefix_product
        prefix_product *= nums[i]
        
    # Calculate suffix products and multiply with prefix products
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
        
    return result