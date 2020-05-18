# ActiveShiftLayer-Naver
Pytorch implementation of "Constructing Fast Networkthrough Deconstruction of Convolution" (https://papers.nips.cc/paper/7835-constructing-fast-network-through-deconstruction-of-convolution.pdf).

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
opASL.cpp is the c++ file containing a custom op. It follows the mathematical formulas in the paper, with a forward function that returns a tensor with ASL updated values in formula 11.
ASL.py supposedly imports the op and defines a new autograd function based on ASL.

# TODO
1. Solve error above
Possibly due to path error from python3 installed in anaconda. 
2. Further error removal
3. Test code efficiency (compare and contrast with paper results)

# 후기
1년간 개인 프로젝트를 위해 Pytorch를 사용하면서 제가 직접 op를 디자인할 기회는 없었는데, 이번에 논문을 읽고 C++로 평소에 생각 없이 쓰던 autograd 함수들을 구현하게 되어 정말 재밌었습니다. 

아직은 서투른지라 코드에 에러가 많지만, 정말 많은 것들을 배울 수 있었던 기회였습니다. 논문을 읽으면서 최근에 배웠던 neural network와 선형대수 개념들을 복습하고 단순한 인공지능 사용자가 아닌 인공지능을 전공하고 발전시키는 연구자로서의 꿈을 다시 확인 할 수 있었던 것 같습니다.

인공지능에서 이렇게 효울성과 정확성을 높이기 위한 다양한 기법들이 쓰이고 있다는 것이 재미있고, 너무나도 빠르게 성장 중인 학문에 입문한다는 것이 설렙니다.  
