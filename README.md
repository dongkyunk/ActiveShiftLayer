# ActiveShiftLayer
Pytorch implementation of "Constructing Fast Networkthrough Deconstruction of Convolution" 
(https://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution.pdf).

# Implementation
The program could not be compiled/tested on my local computer due to an unknown error : 
```bash
"Error checking compiler version for cl"
```
However, the code has all the fundamental details for ASL implementation, and I would be happy to receive further feedback.

The intended code is supposed to be used with the following code: 
```bash
~/setup.py/ install (to install custom C++ op)
```
opASL.cpp is the c++ file containing a custom op. It follows the mathematical formulas in the paper, with a forward function that returns a tensor with ASL updated values in formula 11. (Read comments for detail)

ASL.py supposedly imports the op and defines a new autograd function based on ASL.

# TODO
1. Solve error above

Possibly due to path error from python3 installed in anaconda. 

2. Further error removal

The backward function may be prone to error.

3. Test code efficiency (compare and contrast with paper results) 
