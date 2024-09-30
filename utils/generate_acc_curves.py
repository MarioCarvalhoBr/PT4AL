import numpy as np
import matplotlib.pyplot as plt

with open('main_best.txt', 'r') as f:
    main_best = f.readlines()
    

with open('main_random_best.txt', 'r') as f:
    main_random_best = f.readlines()

scores1 = []
for acc in main_best:
    acc = acc.split()
    scores1.append(float(acc[1]))

scores2 = []
for acc in main_random_best:
    acc = acc.split()
    scores2.append(float(acc[1]))

plt.plot(scores1, label='pt4al')
plt.plot(scores2, label='random')
plt.legend()
plt.title('PT4AL vs Random (+ 50 samples per cycle)')

plt.savefig('pt4al_vs_random.png')
