// AnagramTree class for Project 3 BSTs and Anagrams
// Name: Henry Chien

import java.util.*;
import java.io.*;

public class AnagramTree {
    protected BST<String> tree;

    public AnagramTree(String filename, int maxLength) {
        tree=new BST<String>();
        Scanner sc=getFileScanner(filename);
        while(sc.hasNext()){
            String word=sc.next();
            if(word.length()>maxLength){
                continue;
            }
            char[]wordArray=word.toCharArray();
            Arrays.sort(wordArray);
            String sortedWord=String.valueOf(wordArray);
            if(tree.isEmpty()||tree.find(sortedWord)==null){
                tree.add(sortedWord, word);
            }
            else{
                tree.find(sortedWord).add(word);
            }
        }
    }

    public boolean isEmpty() {
        return tree==null;
    }

    public ArrayList<String> findMatches(String searchKey) {
        char[]wordArray=searchKey.toCharArray();
        Arrays.sort(wordArray);
        searchKey=String.valueOf(wordArray);
        return tree.find(searchKey);
    }
    
    public static Scanner getFileScanner(String filename) {
        Scanner myFile;
        try { myFile = new Scanner(new FileReader(filename)); }
        catch (Exception e) {
            System.out.println("File not found: " + filename);
            return null;
        }
        return myFile;
    }
}