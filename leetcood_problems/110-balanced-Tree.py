# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    #def hight(self,root):
    #   if root == None:
    #       return 0
    #   return max(self.hight(root.left),self.hight(root.right))+1
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def height(node):
            if not node:
                return 0
            left = height(node.left)
            if left == -1:
                return -1
            right = height(node.right)
            if right == -1:
                return -1
            if abs(left-right) > 1:
                return -1
            return max(left, right)+1

        return height(root) != -1

       #if root == None:
       #    return True
       #if abs(self.hight(root.left)-self.hight(root.right)) <=1:
       #    return self.isBalanced(root.left) and self.isBalanced(root.right)
       #else:
       #    return False

       #This commented algorithm is way slower than the one above, becasue it sa
       #saves every nodes height information
