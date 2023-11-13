
#(n-1)*(n-2)*.....
def gamma(x):
    ans = 1
    for i in range(1, x):
        ans *= i
    return ans

def beta(a, b):
    return ((gamma(a) * gamma(b)) / gamma(a+b))

#n*(n-1)*......
def fact(n):
    ans = 1
    for i in range(1, n+1):
        ans *= i
    return ans

def onlineLearning(file, a, b):
    f = open(file, 'r')
    i = 1
    for line in f.readlines():
        line = line.strip('\n')
        print(f'case {i}:{ line}')
        x = 0
        num = 0
        for n in line:
            num += 1
            if n == '1':
                x += 1
        #likelihood = C(s+f, s)*(x^s)*(1-x)^f
        likehood = (fact(num) / (fact(x)*fact(num-x))) * ((x/num)**(x)) * ((1-(x/num))**(num-x))
        print(f'likelihood : {likehood}')
        print(f'beta prior : a = {a} b = {b}')
        a += x
        b += (num - x)
        print(f'beta posterior : a = {a} b = {b}\n')
        i += 1



test = 'test.txt'
print('case 1 : a=0, b=0')
onlineLearning(test, 0, 0)
print('-----------------------------------------')
print('case 2 : a=10, b=1')
onlineLearning(test, 10, 1)

