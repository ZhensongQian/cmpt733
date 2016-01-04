
# Table of Contents
* [Assignment 1: MLlib Basics](#Assignment-1:-MLlib-Basics)
	* [Overview](#Overview)
	* [Part 1: Scalable Machine Learning](#Part-1:-Scalable-Machine-Learning)
		* [What is Matrix Multiplication?](#What-is-Matrix-Multiplication?)
		* [Task A: Scalable Matrix Multiplication](#Task-A:-Scalable-Matrix-Multiplication)
			* [Task Description](#Task-Description)
		* [Task B: Scalable Matrix Multiplication (Sparse Matrix)](#Task-B:-Scalable-Matrix-Multiplication-%28Sparse-Matrix%29)
			* [Task Description](#Task-Description)


# Assignment 1: MLlib Basics

## Overview

[MLlib](http://spark.apache.org/docs/latest/mllib-guide.html) is one of the four major libraries in Spark. Its mission is to make practical machine learning **scalable** and **easy**. From Lecture 1, you have learnt the basic ideas that how MLlib achieves this mission. Assignment 1 will help you to deepen the understanding through several programming tasks.

## Part 1: Scalable Machine Learning

At the first sight, scalable machine learning (ML) seems to be an easy thing to do because Spark has already provided scalable data processing. That is, if we could re-implement existing ML algorithms using Spark, the ML algorithms would inherit the scalability feature (i.e., scaling out to 100+ machines and dealing with petabytes of data) from Spark for free. 

However, the challenging part is that for an ML algorithms that works well on a single machine, it does not mean that the algorithm can be easily extended to the Spark programming framework. Furthermore, to make the algorithm run fast in a distributed environment, we need to carefully select our design choices (e.g., broadcast or not, dense or sparse representation). In Part 1, we will use *Matrix Multiplication* as an example to illustrate these points. 

### What is Matrix Multiplication?

[Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication#Matrix_product_.28two_matrices.29) is a basic operation used by many machine learning algorithms. We consider a special type of matrix multiplication: $A^T A$, where $A$ is a $n\times d$ matrix and $A^T$ is the transpose of $A$. 

$A^T A$ will produce a $d\times d$ matrix computed as follows:

\begin{equation*}
A^T\times A = 
  \begin{bmatrix}
    a_{11} & a_{21} & a_{31} & \dots  & a_{n1} \\
    a_{12} & a_{22} & a_{32} & \dots  & a_{n2} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    a_{1d} & a_{2d} & a_{3d} & \dots  & a_{nd}
\end{bmatrix}
\times
  \begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1d} \\
    a_{21} & a_{22} & \dots  & a_{2d} \\
    a_{31} & a_{32} & \dots  & a_{3d} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{n1} & a_{n2} & \dots  & a_{nd}
\end{bmatrix}
  =
    \begin{bmatrix}
    \sum_{i=1}^{n}a_{i1}\cdot a_{i1} & \sum_{i=1}^{n}a_{i1}\cdot a_{i2} & \dots  & \sum_{i=1}^{n}a_{i1}\cdot a_{id} \\
    \sum_{i=1}^{n}a_{i2}\cdot a_{i1} & \sum_{i=1}^{n}a_{i2}\cdot a_{i2} & \dots  & \sum_{i=1}^{n}a_{i2}\cdot a_{id} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sum_{i=1}^{n}a_{id}\cdot a_{i1} & \sum_{i=1}^{n}a_{id}\cdot a_{i2} & \dots  & \sum_{i=1}^{n}a_{id}\cdot a_{id}
\end{bmatrix}
  .
\end{equation*}

Once you understand the equation, it's actually quite easy to write a python program to compute $A^T A$. Here is the code: 

```python
def matrixMultiply(A):
    n = len(A)
    d = len(A[0])
    
    # result is a dxd matrix
    result = [[0 for i in range(d)] for j in range(d)]

    # iterate through columns of A
    for i in range(d):
        #iterate through columns of A
        for j in range(d):
            result[i][j] = sum([A[k][i]*A[k][j] for k in range(n)])
    return result
    
# 4x2 matrix
A = [[1, 2],
     [3, 4],
     [5, 6],
     [7, 8]]

print matrixMultiply(A)
# Output: 
# [[84, 100], [100, 120]]
```

Intuitively, the algorithm enumerates every two columns of $A$, and then computes the inner product of the two columns. 

### Task A: Scalable Matrix Multiplication

In your first programming task, you will deal with a case that the matrix $A$ has a big $n$ and a small $b$ (e.g., $n=10^9, b=10$). In this case, the matrix can not be stored in a single machine, so you have to distribute the storage. Think about how to implement the `matrixMultiply` function using Spark? 

Please note that if you still use the same algorithm (i.e., enumerating every two columns of $A$ and then computing their inner product), it will be very inefficient because to compute every inner produce, you have to scan the entire data and shuffle a column. See the Spark code below. 

```python
    for i in range(d):
        for j in range(d):
            # Let rddA denote an RDD that represents the matrix A
            result[i][j] = rddA.map(lambda row: row[i] *row[j]).reduce(lambda x, y: x+y)   
```

This example tells us that <font color="blue">_an algorithm that works well in a single machine does not mean that it can be easily extended to the Spark framework_</font>. So you have to be very clever with the distributed implementation.

#### Task Description

<u> Input</u>: You will be given a file of the matrix $A$. The file has $n$ lines, and each line has $d=10$ decimal numbers (separated by a space). The input file might be a distribute file, so please use `sc.textFile()` to read the file.

<u> Output </u>: Compute $A^T A$, and output the result as a file. The result will be a $10\times 10$ matrix. The result can be stored in a single machine, so please write it into a local file (use the Python `write` function).  

You task is to write a Spark program called "matrix_multiply.py". Similar to the assignments that you did in CMPT 732, the program has two command line arguments (Python sys.argv): the input and output directories. Those are appended to the command line in the obvious way, so your command will be something like:

```
spark-submit --master <MASTER> matrix_multiply.py /user/<USERID>/matrix_data /user/<USERID>/matrix_result
```

**Dataset:** Download a sample dataset ``matrix_data.zip`` from ?? . Note that the sample data is only for testing purposes. You should ensure your program to be able to work for a much larger n (e.g., $n=10^9$). 



** Hint: ** Unlike the "inner product"-based definition (as shown above), a matrix multiplication can also be expressed in terms of [outer product](https://en.wikipedia.org/wiki/Matrix_multiplication#Outer_product). That is, $A^T A$ is equal to the sum of the outer products of row vectors, i.e.,

$$A^T A = \sum_{i=1}^{n} a_i \otimes a_i,$$

where $a_i$ is the i-th row vector in $A$, and $\otimes$ denotes an outer product of two vectors.

### Task B: Scalable Matrix Multiplication (Sparse Matrix)

As mentioned in the beginning of this  [section](#Part1:-Scalable-Machine-Learning), to develop an efficient distributed algorithm, we need to carefully select our design choices (e.g., broadcast or not, dense or sparse representation). Next, you will see how to use sparse representation to improve the performance of matrix multiplication.

Suppose you want to compute $A^T A$ as before. But unlike the Task A, here the matrix $A$ is very sparse, where most of the elements in the matrix are zero. If you use the same algorithm as before, the computation cost will be $\mathcal{O}(n*d^2)$. In this task, please think about how to reduce the computation cost to $\mathcal{O}(n*s^2)$ via sparse representation, where $s$ is the number of non-zero elements in each row. 

#### Task Description

<u> Input</u>: You will be given a file of the matrix $A$. The file has $n$ lines, and each line represents a row of the matrix. The row is a $d=$<font color=red>100</font> dimentional vector. The vector is very sparse, which is in the format of
```
index1:value1 index2:value2 index3:value3 ...
```

where `index` is the position of a non-zero element, and `value` is the non-zero element. Note that `index` **starts from zero**, so it is in the range of [0, 99]. For example, "0:0.1 2:0.5 99:0.9" represents the vector of "[0.1, 0, 0.5, 0, 0, ... , 0, 0.9]".

<u> Output </u>: Compute $A^T A$, and output the result as a file. The result will be a $100\times 100$ matrix. The result can be stored in a single machine, so please write it into a local file (use the Python `write` function).  

You task is to write a Spark program called "matrix_multiply_sparse.py". The program has two command line arguments (Python sys.argv): the input and output directories.

**Dataset:** Download a sample dataset ``matrix_data_sparse.zip`` from ?? . Note that the sample data is only for testing purposes. You should ensure your program to be able to work for a much larger n (e.g., $n=10^9$). 



**Hints:** 

1. Take a look at [csr_matrix](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html). Use csr_matrix to represent a sparse row vector. 
2. Think about how to compute the outer product of two sparse vectors via the [transpose](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.transpose.html#scipy.sparse.csr_matrix.transpose) and [multiply](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.multiply.html#scipy.sparse.csr_matrix.multiply) methods.
3. Think about how to add up two csr_matrices. 

## Part 2: Machine Learning Pipeline


```python

```
