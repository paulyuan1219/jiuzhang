# 九章算法班2021

### 704. Binary Search

下面这个结果是我看了令狐冲的二分法模板以后写的，貌似不错，好处在于不用太在意+1 -1这类的问题 呵呵



```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        if len(nums) == 0: # better to use if not nums
            return -1
        
        start = 0
        end = len(nums) - 1
        
        while start + 1 < end:
            mid = int(start + (end - start) / 2)
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        
        return -1
```

第一次写的时候，忘记了`mid=int()`了，所以报错，修改好了就好了。

在参考了模板以后，做了细微的修改。最后优化好的代码如下

```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        if not nums: # if nums is None or len(nums) == 0
            return -1
        
        start = 0
        end = len(nums) - 1
        
        while start + 1 < end: # start, something, end
            # mid = int(start + (end - start) / 2)
            # since python will not overflow, we can just use the following
            mid = (start + end) // 2
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        
        # here, start and end are adjacent
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        
        return -1
```



2021-02-09 lintcode里面重新写了一次，没问题

```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def findPosition(self, nums, target):
        # write your code here
        if not nums:
            return -1
            
        
        start, end = 0, len(nums)-1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                return mid
            
            if nums[mid] < target:
                start = mid
            else:
                end = mid
        
        if nums[start] == target:
            return start
        
        if nums[end] == target:
            return end
        
        return -1
```



### 34. Find First and Last Position of Element in Sorted Array

这道题也是用经典的二分法，简单思考一下，我发现似乎不能够同时寻找first 和 last。所以感觉上一种合适的方式是分开做，简单来说，寻找first 和 last的代码是高度重复的，可以放在一个子函数中，然后再原来的nums上调用两次就可以了，这样来看最简单

```
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        if not nums:
            return [-1, -1]
        
        start, end = 0, len(nums) -1
        
        if target < nums[start] or target > nums[end]:
            return [-1, -1]
        
        
        def func_find_target(nums, s, e, t, is_first):
            """
            
            """
            while s + 1 < e:
                mid = (s + e) // 2

                if nums[mid] == t:
                    if is_first:
                        e = mid
                    else:
                        s = mid
                elif nums[mid] < t:
                    s = mid
                else:
                    e = mid
            
            # here, s and e are adjacent
            if is_first:
                if nums[s] == t:
                    return s
                
                if nums[e] == t:
                    return e
            else:
                if nums[e] == t:
                    return e

                if nums[s] == t:
                    return s

            return -1
        
        res_first = func_find_target(nums, 0, len(nums) - 1, target, True)
        res_last = func_find_target(nums, 0, len(nums) - 1, target, False)
        return [res_first, res_last]
```

> Runtime: 132 ms, faster than 7.48% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.
>
> Memory Usage: 15.6 MB, less than 11.08% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.



仔细想一想，以上代码似乎还可以做一些优化，用来加快速度，对吧 呵呵。

```
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        if not nums:
            return [-1, -1]
        
        start, end = 0, len(nums) -1
        
        if target < nums[start] or target > nums[end]:
            return [-1, -1]
        
        
        def func_find_target(nums, s, e, t, is_first):
            while s + 1 < e:
                mid = (s + e) // 2

                if nums[mid] == t:
                    if is_first:
                        e = mid
                    else:
                        s = mid
                elif nums[mid] < t:
                    s = mid
                else:
                    e = mid
            
            # here, s and e are adjacent
            if is_first:
                if nums[s] == t:
                    return s
                
                if nums[e] == t:
                    return e
            else:
                if nums[e] == t:
                    return e

                if nums[s] == t:
                    return s

            return -1
        
        res_first = func_find_target(nums, 0, len(nums) - 1, target, True)
        
        if res_first != -1:
            res_last = func_find_target(nums, res_first, len(nums) - 1, target, False)
        else:
            res_last = -1
            
        return [res_first, res_last]
```

事实证明 果然快了一点

> Runtime: 88 ms, faster than 49.09% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.
>
> Memory Usage: 15.5 MB, less than 11.08% of Python3 online submissions for Find First and Last Position of Element in Sorted Array.



### \35. Search Insert Position

套用二分模板貌似很容易，首先让start end收敛到最后，如果找得到target就返回index 否则就比较两个index上的值和target的关系来决定到底在哪里插入

```
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        if not nums:
            return [target]
        
        start, end = 0, len(nums) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        
        if nums[start] == target:
            return start
        
        if nums[end] == target:
            return end
        
        if target < nums[start]:
            return start
        elif target < nums[end]:
            return end
        else:
            return end + 1
```



### \981. Time Based Key-Value Store

直观上来看，依次添加element的时候，timestamp会逐渐增加，所以这是一个递增数列，不用担心乱序。

所以，我们的主要任务是在一个排序数组上，找到<= curr_timestamp的最接近的元素。所以用经典的二分法就可以了 呵呵。

```
class TimeMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.ht = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.ht:
            self.ht[key] = []
        
        self.ht[key].append((value, timestamp))
        

    def get(self, key: str, timestamp: int) -> str:
        
        if key not in self.ht:
            return ""
        
        curr_list = self.ht[key]
        start, end = 0, len(curr_list) - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            
            if curr_list[mid][1] == timestamp:
                end = mid
            elif curr_list[mid][1] < timestamp:
                start = mid
            else:
                end = mid
        
        if curr_list[end][1] <= timestamp: # 第一次写的时候漏了[1] 还是粗心了 呵呵
            return curr_list[end][0]
        elif curr_list[start][1] <= timestamp:
            return curr_list[start][0]
        else:
            return ""
         


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
```





### \702. Search in a Sorted Array of Unknown Size

感觉上不难，首先用2的幂来寻找upper bound 然后运用二分即可

```
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader:
#    def get(self, index: int) -> int:

class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        
        if reader.get(0) == 2147483647:
            return -1
        
        k = 0
        while reader.get(2**k) < target:
            k += 1
        
        # here, we know that reader.get(2**k) >= target
        start = 2 ** (k-1)
        end = start * 2
        
        if reader.get(end) == target:
            return end
        
        while start + 1 < end:
            mid = (start + end) // 2
            if reader.get(mid) == target:
                start = mid
            elif reader.get(mid) < target:
                start = mid
            else:
                end = mid
        
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        else:
            return -1
```

上面这个居然没有通过，有点奇怪，看看

```
TypeError: list indices must be integers or slices, not float
    return self.data[index]
Line 56 in get (Solution.py)
    if reader.get(start) == target:
Line 39 in search (Solution.py)
    ret = Solution().search(reader, param_2)
Line 62 in __helper__ (Solution.py)
    ret = __DriverSolution__().__helper__(
Line 94 in _driver (Solution.py)
    _driver()
Line 108 in <module> (Solution.py)
```

看了一下，貌似看不出问题。但是为了速度考虑，可以在k里面直接使用2的幂次方，按照下面的方式就通过了 呵呵

```
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader:
#    def get(self, index: int) -> int:

class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        
        if reader.get(0) == 2147483647:
            return -1
        
        k = 1
        while reader.get(k) < target:
            k *= 2
        
        # here, we know that reader.get(k) >= target
        start = k // 2
        end = k
        
        if reader.get(end) == target:
            return end
        
        while start + 1 < end:
            mid = (start + end) // 2
            if reader.get(mid) == target:
                start = mid
            elif reader.get(mid) < target:
                start = mid
            else:
                end = mid
        
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        else:
            return -1
        
```

> Runtime: 32 ms, faster than 87.33% of Python3 online submissions for Search in a Sorted Array of Unknown Size.
>
> Memory Usage: 15.3 MB, less than 51.13% of Python3 online submissions for Search in a Sorted Array of Unknown Size.

### \69. Sqrt(x)

直观来看 还是用二分法，所以继续使用上面的模板，貌似很好用啊

```
class Solution:
    def mySqrt(self, x: int) -> int:
        
        if x < 2:
            return x
        
        start, end = 1, x
        
        while start + 1 < end:
            mid = (start + end) // 2
            mid_2 = mid * mid
            mid_plus = (mid + 1) * (mid + 1)
            
            if mid_2 <= x and mid_plus >= x:
                start = mid
            elif mid_2 < x:
                start = mid
            else:
                end = mid
        
        lower_val = start * start
        higher_val = end * end
        
        if lower_val <= x:
            return start
        elif x <= higher_val:
            return start
        else:
            return end
```

### \33. Search in Rotated Sorted Array

继续套用二分法的模板，居然work了 妙啊

这里的核心思想是，二分以后，总有一段区间是有序的，我们只要检查target是否落在这个区间即可。如果是，那么就继续看这个区间。如果不是，那么就看另外一个区间。

```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2
            
            if nums[start] < nums[end]:
                # this interval is sorted, easy piece
                if nums[mid] == target:
                    start = mid
                elif nums[mid] < target:
                    start = mid
                else:
                    end = mid
            else:
                # this interval is rotated
                if nums[start] < nums[mid]: # the first half is sorted
                    if nums[start] <= target <= nums[mid]:
                        end = mid
                    else:
                        start = mid
                else: # the 2nd half is sorted
                    if nums[mid] <= target <= nums[end]:
                        start = mid
                    else:
                        end = mid
        
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        else:
            return -1
```

> Runtime: 28 ms, faster than 99.68% of Python3 online submissions for Search in Rotated Sorted Array.
>
> Memory Usage: 14.6 MB, less than 57.05% of Python3 online submissions for Search in Rotated Sorted Array.



### \74. Search a 2D Matrix

直观来看，感觉就是把2d转化成1d来做就可以了。

注意，这中间的转换可以有多重方式，我一开始用了1d index ==> 2d index　然后取出当前index的val。但是其实这两步可以放在一起

```
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        def func_get_value_by_index(index, matrix, num_rows, num_cols):
            row_index = index // num_rows
            col_index = index % num_cols
            
            return matrix[row_index][col_index] # list index out of range
            
            
        if not matrix:
            return False
        
        # Now matrix has at least one elem, i.e. one row
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        if num_cols == 0:
            return False
        
        start, end = 0, num_rows * num_cols - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            curr_val = func_get_value_by_index(mid, matrix, num_rows, num_cols)
            if curr_val == target:
                start = mid
            elif curr_val < target:
                start = mid
            else:
                end = mid
        
        if func_get_value_by_index(start, matrix, num_rows, num_cols) == target:
            return True
        
        if func_get_value_by_index(end, matrix, num_rows, num_cols) == target:
            return True
        
        return False
```

以上代码居然没有通过

```

```

好吧，其实是1d-->2d index出错了。这个转换只需要用到num_cols，根本用不到row_nums

```
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        def func_get_value_by_index(index, matrix, num_cols):
            row_index = index // num_cols
            col_index = index % num_cols
            
            return matrix[row_index][col_index]
            
            
        if not matrix:
            return False
        
        # Now matrix has at least one elem, i.e. one row
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        if num_cols == 0:
            return False
        
        start, end = 0, num_rows * num_cols - 1
        
        while start + 1 < end:
            mid = (start + end) // 2
            curr_val = func_get_value_by_index(mid, matrix, num_cols)
            if curr_val == target:
                start = mid
            elif curr_val < target:
                start = mid
            else:
                end = mid
        
        if func_get_value_by_index(start, matrix, num_cols) == target:
            return True
        
        if func_get_value_by_index(end, matrix, num_cols) == target:
            return True
        
        return False
        
```



### \240. Search a 2D Matrix II

这道题之前看过答案，所以基本上是背下来的。

基本想法就是每次从左往右的水平方向到底，然后从上到下，这是一个排序数组，所以拐角的地方就可以看做mid 每次比较这个拐角的值，可以去掉一行或者一列。这样总体就是往左下方走。一旦找到了target就立即返回，否则就继续走，直到index变成越界，此时就是找不到

```
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        if not matrix:
            return False
        
        num_rows = len(matrix)
        num_cols = len(matrix[0])
        
        row_index, col_index = 0, num_cols - 1
        
        while row_index < num_rows and col_index >= 0:
            # with this condition, it means it's still a valid index
            curr_val = matrix[row_index][col_index]
            
            if curr_val == target:
                return True
            elif curr_val < target:
                row_index += 1
            else:
                col_index -= 1
        
        return False
```



### \875. Koko Eating Bananas

一开始没有想到二分法，但是既然花花这么归类了，所以顺着这个思路就有了。

最基本的，如果K=max(piles) 那么肯定是可以的，所以在[1, K]中间使用二分法

每次遇到一个mid，需要一个子函数来判断当前这个mid是否能通过测试。这个子函数本身复杂度O(logn)

```
class Solution:
    def minEatingSpeed(self, piles: List[int], H: int) -> int:
        
        def func_validate(piles, H, K):
            """
            Check whether Koko can finish piles using K
            """
            curr_H = 0
            for pile in piles:
                hour_1 = pile // K
                hour_2 = pile % K
                
                if hour_2 == 0:
                    curr_H += hour_1
                else:
                    curr_H += hour_1 + 1
            
            return curr_H <= K
                
                
        if len(piles) == H:
            return max(piles)
        
        K = max(piles)
        start, end = 1, K
        
        while start + 1 < end:
            mid = (start + end) // 2
            tmp_result = func_validate(piles, H, mid)
            if tmp_result:
                end = mid
            else:
                start = mid
        
        if func_validate(piles, H, start):
            return start
        
        if func_validate(piles, H, end):
            return end
```

居然有错 为啥

```
Your input
[3,6,7,11]
8
stdout

Output
6
Expected
4
```

好吧，上面这个错了居然 太粗心

```
            return curr_H <= K 其实应该是<H 然后就通过了

```

> Success
>
> [Details ](https://leetcode.com/submissions/detail/452839583/)
>
> Runtime: 576 ms, faster than 28.14% of Python3 online submissions for Koko Eating Bananas.
>
> Memory Usage: 15.7 MB, less than 22.99% of Python3 online submissions for Koko Eating Bananas.

注意，如果可以用math库的话，上面的代码更加简单

```
        def func_validate(piles, H, K):
            """
            Check whether Koko can finish piles using K
            """
            curr_H = sum([math.ceil(pile/K) for pile in piles])
            
            return curr_H <= H
```



### 5. Longest Palindromic Substring

首先，看了一下令狐冲的初步介绍，尝试用O(n^3)的方法来做，看看python里面能否通过

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        def func_validate(s, start, end):
            """
            check if s[start:end+1] is palindrome
            """
            
            is_valid = True
            
            while start <= end:
                if s[start] == s[end]:
                    start += 1
                    end -= 1
                else:
                    is_valid = False
                    break
                    
            return is_valid
                
            
        N = len(s)
        
        res = 1
        
        for start in range(N):
            for end in range(start, N):
                if func_validate(s, start, end):
                    res = max(res, end - start)
        
        return res
```

上面代码输出错误结果

> Wrong Answer
>
> [Details ](https://leetcode.com/submissions/detail/452852873/)
>
> Input
>
> "babad"
>
> Output
>
> 2
>
> Expected
>
> "bab"

好吧，其实是题目没有看清，最后返回的应该是一个substring，而且长度应该是end-start+1

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        def func_validate(s, start, end):
            """
            check if s[start:end+1] is palindrome
            """
            
            is_valid = True
            
            while start <= end:
                if s[start] == s[end]:
                    start += 1
                    end -= 1
                else:
                    is_valid = False
                    break
                    
            return is_valid
                
            
        N = len(s)
        
        res = 1
        res_start = 0
        res_end = 0
        
        for start in range(N - 1):
            for end in range(start + 1, N):
                if func_validate(s, start, end):
                    if res < end - start + 1:
                        res = end - start + 1
                        res_start = start
                        res_end = end
        
        return s[res_start:res_end+1]
```

以上代码应该是正确的，但是超时了，不太好。

以上代码还可以稍微进行优化，就是把长度从大到小来遍历，这样的话只要一遇到第一个substring满足条件，就可以返回了

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        def func_validate(s, start, end):
            """
            check if s[start:end+1] is palindrome
            """
            
            is_valid = True
            
            while start <= end:
                if s[start] == s[end]:
                    start += 1
                    end -= 1
                else:
                    is_valid = False
                    break
                    
            return is_valid
                
        if not s:
            return ""
        
        N = len(s)
        
        for length in range(N, 0, -1):
            for start in range(N + 1 - length):
                if func_validate(s, start, start + length - 1):
                    return s[start: start + length]
```

仍然超时 呵呵，不太好。



根据讲座，使用中心开花发，这样把复杂度变成了O(n^2) 这样就通过了

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        def func_expand(s, start_left, start_right):
            """
            expand the substring from the start positions
            if start_left == start_right, we have a mid elem, then expand it, so the substring length is odd
            if start_left < start_right, we don't have a mid elem, then the substring length is even
            """
            
            while start_left >= 0 and start_right < len(s):
                # we have to make sure both positions are valid
                if s[start_left] == s[start_right]:
                    start_left -= 1
                    start_right += 1
                else:
                    break
            
            # here, we know that s[start_left:start_right+1] is not valid
            return start_left + 1, start_right - 1
            
        if not s:
            return ""
        
        res = 1
        res_left = 0
        res_right = 0
        
        for i in range(len(s) - 1):
            tmp_left, tmp_right = func_expand(s, i, i)
            
            if tmp_right - tmp_left + 1 > res:
                res = tmp_right - tmp_left + 1
                res_left, res_right = tmp_left, tmp_right
            
            tmp_left, tmp_right = func_expand(s, i, i + 1)
            
            if tmp_right - tmp_left + 1 > res:
                res = tmp_right - tmp_left + 1
                res_left, res_right = tmp_left, tmp_right
        
        return s[res_left : res_right + 1]

```

```
Success
Details 
Runtime: 1052 ms, faster than 57.53% of Python3 online submissions for Longest Palindromic Substring.
Memory Usage: 14.5 MB, less than 25.14% of Python3 online submissions for Longest Palindromic Substring.
```

下面，根据讲解内容，实现一下动态规划

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        if not s:
            return ""
        
        N = len(s)
        
        matrix_palindrome = [[False] * N for _ in range(N)]
        
        for i in range(N):
            matrix_palindrome[i][i] = True
        
        for i in range(1, N):
            matrix_palindrome[i][i - 1] = True
        
        res = 1
        res_start = 0
        res_end = 0
        
        for length in range(2, N + 1):
            for start in range(N - length + 1):
                end = start + length - 1
                matrix_palindrome[start][end] = matrix_palindrome[start + 1][end - 1] and (s[start] == s[end])
                if matrix_palindrome[start][end] and length > res:
                    res = length
                    res_start, res_end = start, end
        
        return s[start][end]
```

有Runtme Error: string index out of range in return s[start][end]

好吧，最后一行错了，还是粗心，以后要养成习惯

```
        return s[res_start : res_end + 1] # 这样就对了 呵呵

```



### \516. Longest Palindromic Subsequence

这个的思想跟上面一样 也是用DP就行了

```
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        
        if not s:
            return 0
        
        N = len(s)
        matrix_palindrome = [[0] * N for _ in range(N)]
        
        for i in range(N):
            matrix_palindrome[i][i] = 1
        
        res = 1
        
        for length in range(2, N): # it should be range(2, N + 1)
            for start in range(N - length + 1):

                end = start + length - 1
                
                if s[start] == s[end]:
                    matrix_palindrome[start][end] = 2 + matrix_palindrome[start + 1][end - 1]
                else:
                    matrix_palindrome[start][end] = max(matrix_palindrome[start + 1][end], matrix_palindrome[start][end - 1])
                
                if matrix_palindrome[start][end] > res:
                    res = matrix_palindrome[start][end]
        
        return res

```

测试运行没有通过

Wrong Answer

Runtime: 36 ms



> Your input
>
> "bbbab"
>
> stdout
>
> Output
>
> 3
>
> Diff
>
> Expected
>
> 4

研究一下，看看为什么呢？

好吧，length最后应该是n，又是粗心 呵呵



### \28. Implement strStr()

这个是看了讲解以后做的，但是题目的开始条件不是很清楚，尤其是当两个字符串有一个或者两个都为空的时候。这个要当心

```
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        
        if not haystack:
            return -1
        
        M = len(haystack)
        N = len(needle)
        
        if M < N:
            return -1
        
        for i in range(M - N): # it should be M - N + 1
            b_match = True
            
            for l in range(N):
                if haystack[i + l] != needle[l]:
                    b_match = False
                    break
            
            if b_match:
                return i
        
        return -1
```

以上有错

> Wrong Answer
>
> [Details ](https://leetcode.com/submissions/detail/452926340/)
>
> Input
>
> "a"
> "a"
>
> Output
>
> -1
>
> Expected
>
> 0



### \125. Valid Palindrome

这个好像没什么难度，双指针即可。我是直接按照令狐冲的提示来写的，只是要注意，这里的条件判断要当心

```
class Solution:
    def isPalindrome(self, s: str) -> bool:
        
        if not s:
            return True
        
        def func_validate_char(c):
            return c.isdigit() or c.isalpha()
        
        left, right = 0, len(s) - 1
        
        while left < right:
            
            while left < right and not func_validate_char(s[left]):
                left += 1
            
            while left < right and not func_validate_char(s[right]):
                right -= 1
            
            if left < right and s[left].lower() != s[right].lower():
                return False
            
            left += 1
            right -= 1
        
        return True
```



自己重新做了一遍，错了居然

```
class Solution:
    def isPalindrome(self, s: str) -> bool:
        
        def func_is_valid_char(c):
            return isdigit(c) or isalpha(c) # syntax error

        if not s:
            return False
        
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not func_is_valid_char(s[left]):
                left += 1
            
            while left < right and not func_is_valid_char(s[right]):
                right -= 1
            
            # the following conditions are wrong. Think again
            if left < right and s[left] == s[right]:
                left += 1
                right -= 1
            elif left < right and s[left] != s[right]:
                return False
        
        return True
```

仔细想一想，正确答案是有道理的。我们一直使用left<right 当最后找到一对left right的时候，其实此时只有两种情况。1）要么两个相等，我们继续收缩范围继续匹配 2）要么两个字符不想等，此时匹配失败 就不是palindrome了。

好吧，在我刚刚在我现有的代码上修改了一下，也能通过，但是比较丑，而且要注意lower的使用

```
class Solution:
    def isPalindrome(self, s: str) -> bool:
        
        def func_is_valid_char(c):
            return c.isdigit() or c.isalpha()

        if not s:
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not func_is_valid_char(s[left]):
                left += 1
            
            while left < right and not func_is_valid_char(s[right]):
                right -= 1
            
            if left < right and s[left].lower() == s[right].lower():
                left += 1
                right -= 1
            elif left < right and s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1
        
        return True
```

### \680. Valid Palindrome II

视频里面首先提了一下，应该应用双指针+贪心算法。就是首先比较LR，如果一样就继续往中间比较。如果不想等，则希望去掉L或者去掉R后中间还是回文，如果是就继续，如果不是就返回False。根据这点信息，想一想怎么写代码

```
class Solution:
    def validPalindrome(self, s: str) -> bool:
        
        if not s:
            return True
        
        left, right = 0, len(s) - 1
        
        invalid_count = 0
        while left < right and invalid_count < 2:
            if s[left] != s[right]:
                invalid_count += 1
                
                if invalid_count >= 2:
                    return False
                
                if left + 1 < right:
                    if s[left + 1] == s[right]:
                        left += 1
                    elif s[left] == s[right - 1]:
                        right -= 1
                    else:
                        return False
            
            # here, we know that s[left] == s[right]
            left += 1
            right -= 1
        
        return True
            
```

Wrong Answer

```
Wrong Answer
Details 
Input
"aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga"
Output
false
Expected
true
```

OK debug了一下发现问题所在。正如之前所说，当发现left != right的时候，根据贪心算法，我们应该比较left+1 and right, or left and right -1 这两个里面只要有一个match就可以了，而我上面的代码里面，人为设置了一个优先级，其实是不对的 呵呵。

好的，根据上面的这个思想，我想到了额外再包装一个函数，结果work了 呵呵

```
class Solution:
    def validPalindrome(self, s: str) -> bool:

        if not s:
            return True

        def f1(s, left, right, threshold):
            invalid_count = 0
            while left < right and invalid_count < threshold:
                if s[left] != s[right]:
                    invalid_count += 1

                    if invalid_count >= threshold:
                        return False

                    return f1(s, left + 1, right, threshold - 1) or f1(s, left, right - 1, threshold - 1)

                left += 1
                right -= 1

            return True

        return f1(s, 0, len(s) - 1, 2)
```

```
Success
Details 
Runtime: 188 ms, faster than 34.07% of Python3 online submissions for Valid Palindrome II.
Memory Usage: 14.5 MB, less than 93.92% of Python3 online submissions for Valid Palindrome II.
```

看了一下令狐冲的解答，里面用到了两个子函数，一个是用来返回left != right的第一个位置的，另一个用来判断是否是回文。也蛮不错的 实现一下

```
class Solution:
    def validPalindrome(self, s: str) -> bool:

        if not s:
            return True
        
        def func_find_difference(s, left, right):
            while left < right and s[left] == s[right]:
                left += 1
                right -= 1
            
            return left, right
        
        def func_check_palindrome(s, left, right):
            res_left, res_right = func_find_difference(s, left, right)
            return res_left >= right # 好吧 这里应该是res_right 就对了，所以变量名要简单
        
        tmp_left, tmp_right = func_find_difference(s, 0, len(s) - 1)
        
        if tmp_left >= tmp_right:
            return True
        
        return func_check_palindrome(s, tmp_left + 1, tmp_right) or func_check_palindrome(s, tmp_left, tmp_right - 1)
```

居然没有通过

```
Wrong Answer
Details 
Input
"deeee"
Output
false
Expected
true
```



### 1. Two Sum

注意，这道题目在leetcode和令狐冲讲义里面不太一样。这里以leetcode为主。

首先，因为要返回元素的index，所以额外的数据结构应该用哈希表，而不是set。如果按照令狐冲的题目，只需要返回元素的值，那么用set就够了。

其次是关于重复元素的问题，比如[3,3] with target =6, the result is [0, 1] 所以第一反应是，我们的key-value pair should not just be "3-->0", it should be "3-->[0, 1]" 但是仔细想一想，其实没必要，对于上面这个例子，看到第一个3的时候我们已经有ht里面的"3-->0" 所以当我们移动到第二个3的时候，此时当前index = 1,所以我们直接返回当前index以及ht里面存放的过去见过的index即可。如果target用不到两个重复元素的想加，那么我们只需要保存若干重复元素里面的一个index即可，比如[3, 3, 7] with target = 10，此时res = [0, 2] or res=[1,2]都是对的。所以我们不用太在意，直接保存第一个即可。

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        ht = {}
        for i in range(len(nums)):
            curr_val = nums[i]
            curr_val_complement = target - curr_val
            
            if curr_val_complement in ht:
                return i, ht[curr_val_complement]
            
            ht[curr_val] = i
```

```
Success
Details 
Runtime: 44 ms, faster than 85.90% of Python3 online submissions for Two Sum.
Memory Usage: 14.4 MB, less than 47.62% of Python3 online submissions for Two Sum.
```

令狐冲还介绍了排序+双指针的做法，也可以试试看

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        nums.sort()
        left, right = 0, len(nums) - 1
        
        while left < right:
            tmp_sum = nums[left] + nums[right]
            
            if tmp_sum == target:
                return left, right
            elif tmp_sum < target:
                left += 1
            else:
                right -= 1
        
        return False
```

居然有错

```
Wrong Answer
Details 
Input
[3,2,4]
6
Output
[0,2]
Expected
[1,2]
```

好吧，其实是这样，重排序后，原来的序号丢了。。。所以报错。所以我们需要重排序原来的tuple

这里的核心难点在于如何排序一组tuple，好好掌握一下

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        nums_tuples = [ (num, index) for index, num in enumerate(nums)]
        nums_tuples.sort()
        left, right = 0, len(nums_tuples) - 1
        
        while left < right:
            tmp_sum = nums_tuples[left][0] + nums_tuples[right][0]
            
            if tmp_sum == target:
                return nums_tuples[left][1], nums_tuples[right][1]
            elif tmp_sum < target:
                left += 1
            else:
                right -= 1
        
        return [-1, -1]
```



### \1099. Two Sum Less Than K

这道题算是上面的follow up，参考一下改怎么搞

```
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        
        nums.sort()
        
        left, right = 0, len(nums) - 1
        
        res = -1
        
        while left < right:
            
            curr_sum = nums[left] + nums[right]
            
            if curr_sum == k:
                return k
            elif curr_sum < k:
                if curr_sum > res:
                    res = curr_sum
                left += 1
            else:
                right -= 1
        
        return res
```

以上没通过，其实是因为审题不清，因为题目要求严格小于target，不能够有等于 呵呵

```
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        
        nums.sort()
        
        left, right = 0, len(nums) - 1
        
        res = -1
        
        while left < right:
            
            curr_sum = nums[left] + nums[right]
            
            if curr_sum < k:
                if curr_sum > res:
                    res = curr_sum
                left += 1
            else:
                right -= 1
        
        return res
```

```
Success
Details 
Runtime: 44 ms, faster than 67.41% of Python3 online submissions for Two Sum Less Than K.
Memory Usage: 14.4 MB, less than 23.95% of Python3 online submissions for Two Sum Less Than K.
```

2020-02-13 又做了一次，没问题

```
class Solution:
    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        
        nums.sort()
        
        left, right = 0, len(nums) - 1
        
        res = -1
        while left < right:
            tmp_sum = nums[left] + nums[right]
            
            if tmp_sum == k:
                right -= 1
            elif tmp_sum < k:
                res = max(res, tmp_sum)
                left += 1
            else:
                right -= 1
        
        return res
```





### \167. Two Sum II - Input array is sorted

没啥难的，已经排序好了，直接用双指针就可以。

就是i要注意，最后返回的结果是以1为开始的，所以要加一就行了

```
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            curr_sum = numbers[left] + numbers[right]
            if curr_sum == target:
                return left + 1, right + 1
            elif curr_sum < target:
                left += 1
            else:
                right -= 1
        
        return [-1, -1]
```



### \912. Sort an Array

首先用快速排序实现一把，基本就按照令狐冲的视频讲座来实现，注意大于号等于号的使用

```
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def quickSort(nums, start, end):
            if start >= end:
                return
            
            left = start
            right = end
            
            pivot = nums[(left + right) // 2] # value, not index
            
            while left <= right: # <=, not <
                while left <= right and nums[left] < pivot: # < pivot, not <= pivot
                    left += 1
                
                while left <= right and nums[right] > pivot:
                    right -= 1
                
                if left <= right:
                    nums[left], nums[right] = nums[right], nums[left]
                    
                    left += 1
                    right -= 1
            
            quickSort(nums, start, right)
            quickSort(nums, left, end)
        
        quickSort(nums, 0, len(nums) - 1)
        return nums
```

实现的时候要注意三个要点，都在注释里面了，主要是为了提高效率。防止当数组中存在大量重复元素的时候，比如[1,1,1...,1,1,1,2] 此时 partition会一直走到底，不太好，不平衡

接下来，实现一把mergeSort 首先按照令狐冲的思路写一下看看

```
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def mergeSort(nums, start, end, res):
            if not nums or start >= end:
                return
            
            # here, we know nums is valid array and start < end
            mergeSort(nums, start, (start + end) // 2, res)
            mergeSort(nums, (start + end) // 2 + 1, end, res)
            merge(nums, start, end, res)
        
        def merge(nums, start, end, res):
            mid = (start + end) // 2
            
            left_index = start
            right_index = mid + 1
            res_index = left_index
            
            while left_index <= mid and right_index <= end:
                if nums[left_index] <= nums[right_index]:
                    res[res_index] = nums[left_index]
                    left_index += 1
                else:
                    res[res_index] = nums[right_index]
                    right_index += 1
                res_index += 1
            
            while left_index <= mid:
                res[res_index] = nums[left_index]
                left_index += 1
                res_index += 1
                
            while right_index <= end:
                res[res_index] = nums[right_index]
                right_index += 1
                res_index += 1
            
            for _ in range(start, end + 1):
                nums[_] = res[_]
        
        res = [0] * len(nums)
        return mergeSort(nums, 0, len(nums) - 1, res)
```

没有通过 有错误

> Wrong Answer
>
> Runtime: 40 ms
>
> 
>
> Your input
>
> [5,2,3,1]
>
> stdout
>
> Output
>
> []
>
> Diff
>
> Expected
>
> [1,2,3,5]

好吧，其实就是最后一句出错了，因为我们所有的结果都是放在nums里面的，直接返回nums即可

```
        mergeSort(nums, 0, len(nums) - 1, res)
        return nums
```



### \215. Kth Largest Element in an Array

这个有意思，必须掌握，看了令狐冲的视频以后先实现一下。

```
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        def quickSelect(nums, start, end, k):
            # from nums[start : end + 1], select out the k-th largest element
            if start >= end:
                return nums[start]
            
            left = start
            right = end
            mid = (start + end) // 2
            pivot = nums[mid]
            
            while left <= right:
                while left <= right and nums[left] > pivot:
                    left += 1
                
                while left <= right and nums[right] < pivot:
                    right -= 1
                
                while left <= right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1
            
            # at this point, we have 3 intervals
            # nums[start]...nums[right], all element here >= pivot, total_count = right - start + 1
            # nums[right + 1]
            # nums[left]...nums[end], all elements here <= pivot
            
            if k <= right - start + 1:
                return quickSelect(nums, start, right, k)
            
            if k > left - start:
                return quickSelect(nums, left, end, k - (left - start))
            
            return nums[right + 1]
        
        return quickSelect(nums, 0, len(nums) - 1, k)
```

没有通过 

> Wrong Answer
>
> Runtime: 36 ms
>
> 
>
> Your input
>
> [3,2,1,5,6,4]
> 2
>
> stdout
>
> Output
>
> 3
>
> Diff
>
> Expected
>
> 5

把以下代码放在pycharm里面调试看一下

```
class Solution:
    def findKthLargest(self, nums, k):

        def quickSelect(nums, start, end, k):
            # from nums[start : end + 1], select out the k-th largest element
            if start >= end:
                return nums[start]

            left = start
            right = end
            mid = (start + end) // 2
            pivot = nums[mid]

            while left <= right:
                while left <= right and nums[left] > pivot:
                    left += 1

                while left <= right and nums[right] < pivot:
                    right -= 1

                while left <= right: # it should be if, not while
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1

            # at this point, we have 3 intervals
            # nums[start]...nums[right], all element here >= pivot, total_count = right - start + 1
            # nums[right + 1]
            # nums[left]...nums[end], all elements here <= pivot

            if k <= right - start + 1:
                return quickSelect(nums, start, right, k)

            if k > left - start:
                return quickSelect(nums, left, end, k - (left - start))

            return nums[right + 1]

        return quickSelect(nums, 0, len(nums) - 1, k)

sol1 = Solution()
source = [3, 2, 1, 5, 6, 4]

res = sol1.findKthLargest(source, 2)
print(f"res = {res}")
```

还是一样的结果，debug看看

好吧，就是把一个if语句错写成了while 所以报错，醉了

### \75. Sort Colors

这个感觉上就是快速排序啊，应该不难 看看

```
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        def quickSort(nums, start, end, target):
            
            if start >= end:
                return
            
            left = start
            right = end
            
            while left <= right:
                while left <= right and nums[left] < target:
                    left += 1
                
                while left <= right and nums[right] >= target:
                    right -= 1
                
                if left <= right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1
            
            return left, right
        
        
        l1, r1 = quickSort(nums, 0, len(nums) -1, 1)
        l2, r2 = quickSort(nums, 0, len(nums) -1, 2)
        return
```

居然错误，好吧，input=[0]的时候出错了，抛出异常

```
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        def quickSort(nums, start, end, target):
            
            if start >= end:
                return
            
            left = start
            right = end
            
            while left <= right:
                while left <= right and nums[left] < target:
                    left += 1
                
                while left <= right and nums[right] >= target:
                    right -= 1
                
                if left <= right:
                    nums[left], nums[right] = nums[right], nums[left]
                    left += 1
                    right -= 1
            
            return left, right
        
        if not nums or len(nums) == 1:
            return
        
        l1, r1 = quickSort(nums, 0, len(nums) -1, 1)
        l2, r2 = quickSort(nums, 0, len(nums) -1, 2)
        return
```

上面这个代码就对了，没问题。注意，因为只有0 1 2这三个数，所以我们不需要像快速排序一样做递归，而只要用不同的target or threshold做两次循环即可，不是很难。

### 15. 3Sum

第一眼看到，因为我们需要全部都是自然数，而且a+b+c=0，所以以前我会先排序，然后按照大于零小于零的情况来分别处理。但是这样其实很复杂，容易出错。

看了一眼视频的开头，其实本质上就是dc模式，把3sum==>2sum即可，先想一想该怎么做。

首先已知条件，2sum已经能够熟练的解决。

OK 套用一个例子就清楚了

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

首先，可以把nums排序，变成`[-4, -1, -1, 0, 1, 2]` 同时假设`a<=b<=c`。 所以最开始a=-4, 我们希望在剩下的里面寻找b+c=4。然后，a=-1，我们希望在右侧寻找b+c=1。注意，这时候我们只需要关注a=-1的右侧，而不用管a=-1的左边，就是-4这个元素。因为按照我们之前的假设，a=-4这个情况已经被考虑过了，所以这里不用重复考虑 呵呵。

首先排序需要O(nlogn) 然后对a遍历需要O(n), within each loop 我们需要双指针，所以还是O(n) 总体就是O(n^2)

思路有了，下面写一下吧。

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def twoSum(nums, start, end, target, set_res):
            # assert len(nums) >= 2
            if start >= end:
                # less than 2 elems
                return
            
            # There are at least 2 elems in nums
            left, right = start, end
            while left < right:
                curr_sum = nums[left] + nums[right]
                
                if curr_sum == target:
                    tmp_key = (-target, nums[left], nums[right])
                    set_res.add(tmp_key)
                    twoSum(nums, left + 1, right, target, set_res)
                    twoSum(nums, left, right + 1, target, set_res)
                elif curr_sum < target:
                    left += 1
                else:
                    right -= 1
            
            return
        
        if not nums or len(nums) < 3:
            return []
        
        set_res = set([])
        nums.sort()
        
        for i in range(len(nums) - 2):
            twoSum(nums, i, len(nums) - 1, -nums[i], set_res)
        
        return list(set_res)
```

汗，好像遇到死循环了 呵呵

```
RecursionError: maximum recursion depth exceeded in comparison
    set_res.add(tmp_key)
Line 16 in twoSum (Solution.py)
  [Previous line repeated 996 more times]
    twoSum(nums, left, right + 1, target, set_res)
Line 18 in twoSum (Solution.py)
    twoSum(nums, left, right + 1, target, set_res)
Line 18 in twoSum (Solution.py)
    twoSum(nums, left, right + 1, target, set_res)
Line 18 in twoSum (Solution.py)
```



看了一下，是上面的right+1错了 应该是right-1

但是之后造成time limit exceeded

想了一下，可能是重复的代码那里除了问题，改一下，出来新的错误

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def twoSum(nums, start, end, target, set_res):
            # assert len(nums) >= 2
            if start >= end:
                # less than 2 elems
                return
            
            # There are at least 2 elems in nums
            left, right = start, end
            while left < right:
                curr_sum = nums[left] + nums[right]
                
                if curr_sum == target:
                    tmp_key = (-target, nums[left], nums[right])
                    set_res.add(tmp_key)
                    left += 1
                    right -= 1
#                    twoSum(nums, left + 1, right, target, set_res)
#                    twoSum(nums, left, right - 1, target, set_res)
                elif curr_sum < target:
                    left += 1
                else:
                    right -= 1
            
            return
        
        if not nums or len(nums) < 3:
            return []
        
        set_res = set([])
        nums.sort()
        
        for i in range(len(nums) - 2):
            twoSum(nums, i, len(nums) - 1, -nums[i], set_res)
        
        return list(set_res)
```

Wrong Answer:

> Wrong Answer
>
> [Details ](https://leetcode.com/submissions/detail/455624463/)
>
> Input
>
> [1,2,-2,-1]
>
> Output
>
> [[-1,-1,2]]
>
> Expected
>
> []

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def twoSum(nums, start, end, target, set_res):
            # assert len(nums) >= 2
            if start >= end:
                # less than 2 elems
                return
            
            # There are at least 2 elems in nums
            left, right = start, end
            while left < right:
                curr_sum = nums[left] + nums[right]
                
                if curr_sum == target:
                    tmp_key = (-target, nums[left], nums[right])
                    set_res.add(tmp_key)
                    left += 1
                    right -= 1
#                    twoSum(nums, left + 1, right, target, set_res)
#                    twoSum(nums, left, right - 1, target, set_res)
                elif curr_sum < target:
                    left += 1
                else:
                    right -= 1
            
            return
        
        if not nums or len(nums) < 3:
            return []
        
        set_res = set([])
        nums.sort()
        
        for i in range(len(nums) - 2):
            twoSum(nums, i + 1, len(nums) - 1, -nums[i], set_res)
        
        return list(set_res)
            
```

好吧，研究了一下上面的错误，就是每次选择右侧的时候粗心了，应该是选择右侧，不能包括当前的元素a 呵呵

