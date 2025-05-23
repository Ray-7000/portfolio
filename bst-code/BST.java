import java.util.ArrayList;

// BST class for Project 3 BSTs and Anagrams
// Name: Henry Chien

public class BST<E extends Comparable<E>> {
    private TreeNode root;
    private static boolean balance=true;
    
    public BST()     {
        root = null;
    }
    
    public boolean isEmpty()     {
        return root == null;
    }
    
    @Override
    public boolean equals(Object tree) {
        if (this == tree) return true;
        if (tree == null || !(tree instanceof BST<?>)) return false;
        BST<?> otherTree = BST.class.cast(tree);
        return equals(this.root, otherTree.root);
    }

    private boolean equals(TreeNode root1, BST<?>.TreeNode root2) {
        if(root1==root2){
            return true;
        }
        if(root1==null||root2==null){
            return false;
        }
        return root1.data.equals(root2.data) && equals(root1.left, root2.left) && equals(root1.right, root2.right);
    }
    
    public void add(E value) {
        root = addHelper(root, value);
    }

    public void add(E sorted, E unsorted){
        root=addHelper(root, sorted);
        find(sorted).add(unsorted);
    }
    
    private TreeNode addHelper(TreeNode rt, E value) {
        if (rt == null)
            return new TreeNode(value, null, null);
        
        if (value.compareTo(rt.data) < 0)
            rt.left = addHelper(rt.left, value);
        else if (value.compareTo(rt.data) > 0)
            rt.right = addHelper(rt.right, value);
        else
            throw new IllegalStateException("Duplicate value in tree " + value);
        
        return rt;  
    }

    public int height(){
        return height(root);
    }

    private int height(TreeNode rt){
        if(rt==null){
            return -1;
        }
        int heightLeft=height(rt.left);
        int heightRight=height(rt.right);
        return 1+Math.max(heightLeft, heightRight);
    }

    public boolean isBalanced(){
        isBalanced(root);
        return balance;
    }

    private void isBalanced(TreeNode rt){
        if(rt!=null){
            int diff=height(rt.left)-height(rt.right);
            if(Math.abs(diff)>=2){
                balance=false;
            }
            else{
                isBalanced(rt.left);
                isBalanced(rt.right);
            }
        }
    }

    public void inOrder() {
        inOrder(root);
    }
    
    private void inOrder(TreeNode rt) {
        if (rt != null) {
            inOrder(rt.left);
            System.out.print(rt.data + " ");
            inOrder(rt.right);
        }
    }
    
    public void preOrder() {
        preOrder(root);
    }
    
    private void preOrder(TreeNode rt) {
        if (rt != null) {
            System.out.print(rt.data + " ");
            preOrder(rt.left);
            preOrder(rt.right);
        }
    }
    
    public int size() {
        return size(root);
    }
    
    private int size(TreeNode rt) {
        if (rt == null)
            return 0;
        return 1 + size(rt.left) + size(rt.right);
    }
    
    public boolean contains(E value) {
        TreeNode rt = root;
        while (rt != null) {
            if (value.compareTo(rt.data) == 0)
                return true;
            else if (value.compareTo(rt.data) < 0)
                rt = rt.left;
            else
                rt = rt.right;
        }
        return false;
    }

    // returns a String that prints tree top to bottom, right to left in a 90-degree rotated level view
    public String toString() {
        StringBuilder result =  new StringBuilder();
        return toString(result, -1, root).toString();
    }
    
    public StringBuilder toString(StringBuilder res, int height, TreeNode rt) {
        if (rt != null) {
            height++;
            res = toString(res, height, rt.right);
            for (int i = 0; i < height; i++)
                res.append("\t");
            res.append(rt.data + "\n");
            res = toString(res, height, rt.left);
        }
        return res;
    }

    public ArrayList<E> find(E word){
        TreeNode rt = root;
        while (rt != null) {
            if (word.compareTo(rt.data) == 0)
                return rt.words;
            else if (word.compareTo(rt.data) < 0)
                rt = rt.left;
            else
                rt = rt.right;
        }
        return null;
    }
    
    // The TreeNode class is a private inner class used (only) by the BST class
    private class TreeNode {
        private E data;
        private TreeNode left, right;
        private ArrayList<E> words;
        
        private TreeNode(E data, TreeNode left, TreeNode right) {
            this.data = data;
            this.left = left;
            this.right = right;
            words=new ArrayList<E>();
        }
    }
    
    public static void main(String[] args) {
        BST<Integer> treeTest = new BST<Integer>();
        BST<Integer> treeTest2 = new BST<Integer>();
        treeTest2.add(7);
        treeTest2.add(11);
        treeTest2.add(4);
        treeTest2.add(10);
        treeTest2.add(6);
        treeTest2.add(8);

        treeTest.add(7);
        treeTest.add(5);
        treeTest.add(4);
        treeTest.add(10);
        treeTest.add(6);
        treeTest.add(8);
        /*treeTest.inOrder();
        System.out.println();
        treeTest.preOrder();
        System.out.println();
        System.out.println(treeTest.size());
        System.out.println();
        System.out.println(treeTest.contains(6));
        System.out.println(treeTest.contains(112));
        System.out.println(treeTest.contains(7));
        System.out.println(treeTest.contains(10));
        System.out.println();*/
        System.out.println("Height: "+treeTest.height());
        System.out.println();
        System.out.println("Tree is Balanced: "+treeTest.isBalanced());
        System.out.println();
        System.out.println("2 trees are equal: "+treeTest.equals(treeTest2));
        System.out.println();
        System.out.println(treeTest);
    }
    
}