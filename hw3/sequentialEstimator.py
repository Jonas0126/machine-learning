from gaussianGenerator import *




#mu_next = ((new_point-mu_n)/n+1) + mu_n
def update_mu(old_mu, new_p, n):
    return (old_mu + ((new_p-old_mu)/n))

#var_next = (((new_p-old_mu)*(new_p-new_mu)-old_var)/n) + old_var
def update_variance(old_mu, new_mu, old_variance, new_p, n):
    return (((new_p-old_mu)*(new_p-new_mu)-old_variance)/n) + old_variance


if __name__ == '__main__':

    mean = float(input('mean : '))
    variance = float(input('variance : '))
    old_mu = 0
    old_var = 0
    new_mu = 0
    new_var = 0
    n = 0
    while(1):
        new_p = gaussian(mean, variance)
        print(f'add data point : {new_p}')
        n += 1
        old_mu = new_mu
        old_var = new_var
        new_mu = update_mu(old_mu, new_p, n)
        new_var = update_variance(old_mu, new_mu, old_var, new_p, n)
        print(f'mean : {new_mu}    variance : {new_var}')
        if abs(mean-new_mu) <= 0.01 and abs(variance-new_var) <= 0.01:
            break
    
