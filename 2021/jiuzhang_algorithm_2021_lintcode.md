# Lintcode题解

### 经典二分查找问题

套用二分模板 没有任何问题

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



### 目标最后位置

套用模板，居然出错？

```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def lastPosition(self, nums, target):
        # write your code here
        if not nums:
            return -1
            
        
        start, end = 0, len(nums)-1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target: 这里出了问题，因为这里的三个条件应该是并列的，而我这里==条件其实用了两次 呵呵
                start = mid

            if nums[mid] < target: 改成elif就成功了 呵呵
                start = mid
            else:
                end = mid
        
        if nums[end] == target:
            return end
        
        if nums[start] == target:
            return start
        
        return -1
```

Wrong answer

```
[1,2,2,4,5,5]

expected: 5
output: 4
```

好吧，仔细想了一下，是条件分支那里出了问题

### 二分查找

还是套用模板，还是有错 醉了

```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def binarySearch(self, nums, target):
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

输入

查看差别

```
[3,4,5,8,8,8,8,10,13,14]
8
```

输出

```
4
```

期望答案

```
3
```

好吧，第一个 没看清，居然。稍微修改一下就通过了 呵呵

```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def binarySearch(self, nums, target):
        # write your code here
        if not nums:
            return -1
            
        
        start, end = 0, len(nums)-1
        
        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] == target:
                end = mid
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

### 山脉序列中的最大值

首先，用线性方法最简单，先做一下，成功

```
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums:
            return -1
        
        res = nums[0]
        
        for i in range(1, len(nums)):
            if nums[i] < nums[i-1]:
                return nums[i-1]
        
        return nums[-1]
```

然后想一想二分法怎么做呢

感觉上有点搞。理论上找到mid后，再和left right作比较 会有若干中情况

```
start < mid and mid < end 此时明确可知 start < mid < peak > mid ==> [mid, end]
start < mid and mid > end 此时明确可知 start < mid > end 但是peak可以在mid左边或者右边，不确定，不能二分
start > mid and mid < end V字形，不可能
start > mid and mid > end 此时明确可知 start < peak > mid > end ==> [start, mid]

还没考虑start, end互相之间的关系，有点搞
```

好像有点卡住

看了答案 好吧 直接比较mid mid+1即可。醉了

```
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums:
            return -1
        
        start, end = 0, len(nums) -1
        
        while start + 1 < end:
            mid = (start + end) // 2
            
            if nums[mid] < nums[mid+1]:
                start = mid
            else:
                end = mid
        
        return max(nums[start], nums[end])
```





