＃中序遍历 in-order traversal

if root.left==None and root.right ==None:
    print root.val
else:
    self.isSymmetric(root.left)
    print root.val
    self.isSymmetric(root.right)
＃左序遍历 pre-order traversal

if root.left==None and root.right ==None:
    print root.val
else:

    print root.val
    self.isSymmetric(root.left)
    self.isSymmetric(root.right)

＃右序遍历 post-order traversal

if root.left== None and root.right == None:
    print root.val
else:

    self.isSymmetric(root.left)
    self.isSymmetric(root.right)
    print root.val
