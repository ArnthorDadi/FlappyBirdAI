import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def readFile(filename):
    bird_score = []
    f = open(filename+".txt", "r")
    for x in f:
        print(x[1:-2])
        score, episode = x[1:-2].split(",")
        episode = episode[1:]
        score = float(score)
        episode = int(episode)
        bird_score.append({
            'episode': episode,
            'data': score-1
        })
        #if(episode == 5000):
        #    break
    bird_score_df = pd.DataFrame(bird_score)
    return bird_score_df

#folder = 'alpha'
folder = 'discountFactorV2'
#folder = 'epsilon'

#qlearning1 = readFile('./'+ folder +'/0.25/avgScore')
#qlearning2 = readFile('./'+ folder +'/0.5/avgScore')
qlearning3 = readFile('./'+ folder +'/0.75/avgScore')
qlearning4 = readFile('./'+ folder +'/1/avgScore')

plt.title("Q-Learning: Influence of discountFactor on preformance")
plt.xlabel("Nr of training episodes")
plt.ylabel("Avg score of each 100 value episodes")

#plt.plot(qlearning1.episode, qlearning1.data, "o-", color="r", label="0.25")
#plt.plot(qlearning2.episode, qlearning2.data, "o-", color="g", label="0.5")
plt.plot(qlearning3.episode, qlearning3.data, "o-", color="b", label="0.75")
plt.plot(qlearning4.episode, qlearning4.data, "o-", label="1")

plt.legend(loc="upper left")
plt.show()
