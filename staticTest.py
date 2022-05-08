import scipy.stats as st                                                        # knihovna pro statisticke vypocty
import numpy as np                                                              # knihovna pro pokrocilou matematiku

resultFile1 = 'experiments/result1.npy'
resultFile2 = 'experiments/result2.npy'

def main():
    datasetA_min = np.load(resultFile1)
    print(datasetA_min)

    datasetB_min = np.load(resultFile2)
    print(datasetB_min)

    normal = True

    alpha = 0.05
    t, p = st.normaltest(datasetA_min)
    if (p > alpha):
        print('rozdil NENI vyznamny, rozlozeni JE normalni, p = ', p)
    else:
        print('rozdil JE vyznamny, rozlozeni NENI normalni, p = ', p)
        normal = False

    t, p = st.normaltest(datasetB_min)
    if (p > alpha):
        print('rozdil NENI vyznamny, rozlozeni JE normalni, p = ', p)
    else:
        print('rozdil JE vyznamny, rozlozeni NENI normalni, p = ', p)
        normal = False
    
    if not normal:
        t, p = st.mannwhitneyu(datasetA_min,datasetB_min)
        if (p > alpha):
            print('rozdil NENI vyznamny, stredni hodnoty si JSOU podobne, p = ', p)
        else:
            print('rozdil JE vyznamny, stredni hodnoty si NEJSOU podobne, p = ', p)
    else:
        t, p = st.ttest_ind(datasetA_min,datasetB_min)
        if (p > alpha):
            print('rozdil NENI vyznamny, rozlozeni MOHOU mit stejnou streni hodnotu, p = ', p)
        else:
            print('rozdil JE vyznamny, rozlozeni NEMOHOU mit stejnou streni hodnotu, p = ', p)

main()