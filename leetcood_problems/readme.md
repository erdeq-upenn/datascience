Leetcode problems
===
#  Solved Problems
**BinaryTree Problems**

|number|description|level|
|---|---|---|
|[104](https://leetcode.com/problems/maximum-depth-of-binary-tree/)|Maximum depth of binary Tree|E|
~~~
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return max(self.maxDepth(root.left), self.maxDepth(root.right))+1 if root else 0
~~~
|number|description|level|
|---|---|---|
|[111](https://leetcode.com/problems/minimum-depth-of-binary-tree/)|minimum depth of binary Tree|E|
~~~
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if root == None:
            return 0
        elif root.left == None and root.right == None:
            return 1
        elif root.left == None and root.right != None:
            return self.minDepth(root.right)+1
        elif root.right ==None and root.left != None:
            return self.minDepth(root.left)+1
        elif root.left != None and root.right !=None:
            return min(self.minDepth(root.left),self.minDepth(root.right))+1
~~~

|number|description|level|
|---|---|---|
|[110](https://leetcode.com/problems/balanced-binary-tree/)|Balanced binaryTree|E|
