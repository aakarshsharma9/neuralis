from collections import defaultdict, Counter


# 1. Contains Duplicate
def containsDuplicate(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))


# 2. Unique Prime Factors
def get_unique_prime_factors(n: int) -> list[int]:
    factors = []
    if n % 2 == 0:
        factors.append(2)
        while n % 2 == 0:
            n //= 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        factors.append(n)
    return factors


# 3. Fizz Buzz
def fizzBuzz(n: int) -> list[str]:
    res = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            res.append("FizzBuzz")
        elif i % 3 == 0:
            res.append("Fizz")
        elif i % 5 == 0:
            res.append("Buzz")
        else:
            res.append(str(i))
    return res


# 4. Container With Most Water
def maxArea(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        width = right - left
        max_water = max(max_water, width * min(height[left], height[right]))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water


# 5. Valid Palindrome
def isPalindrome(s: str) -> bool:
    filtered = [ch.lower() for ch in s if ch.isalnum()]
    return filtered == filtered[::-1]


# 6. Longest Common Prefix
def longestCommonPrefix(strs: list[str]) -> str:
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


# 7. Convert Sentence to CamelCase
def toCamelCase(s: str) -> str:
    words = s.strip().split()
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


# 8. Longest Palindromic Substring
def longestPalindrome(s: str) -> str:
    def expand(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    res = ""
    for i in range(len(s)):
        res = max(res, expand(i, i), expand(i, i + 1), key=len)
    return res


# 9. Group Anagrams
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())


# 10. Two Sum
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []


# 11. Top K Frequent Elements
def topKFrequent(nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    return [item for item, _ in count.most_common(k)]


# 12. Min Stack
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val_min = val if not self.min_stack else min(val, self.min_stack[-1])
        self.min_stack.append(val_min)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# 13. Intersection of Two Arrays
def intersection(nums1: list[int], nums2: list[int]) -> list[int]:
    return list(set(nums1) & set(nums2))


# 14. Roman to Integer
def romanToInt(s: str) -> int:
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev_val = 0
    for ch in reversed(s):
        curr_val = values[ch]
        if curr_val < prev_val:
            total -= curr_val
        else:
            total += curr_val
            prev_val = curr_val
    return total


# Example usage:
print(get_unique_prime_factors(60))  # Output: [2, 3, 5]
print(get_unique_prime_factors(315))  # Output: [3, 5, 7]
x=containsDuplicate([1,2,2])
print(x)

