# BIN - Optimalizace řešení hry Sudoku genetickým algoritmem

## Zadání
Hra Sudoku je tvořena tabulkou velikosti 9x9 sestávající z devíti menších tabulek (čtverců) velikosti 3x3, částečně předvyplněnou číslicemi of 1 do 9. Cílem hry je volná místa v tabulce doplnit takovým způsobem, aby žádný z řádků, sloupců, ani čtverců neobsahoval stejnou číslici dvakrát. Jedním ze způsobů jak tuto hru řešit je použití genetického algoritmu.

Při řešení se inspirujte metodami řešení problému problému barvení grafu (číslice představují barvy, pole tabulky představují vrcholy, pole na stejném řádku, sloupci a čtverci jsou spojeny hranami) a proveďte sady experimentů využívajících různé varianty genetických operátorů (selekce, křížení atd.).

## Knihovny
- deap
- random
- numpy
- collections
- matplotlib
- seaborn
- networkx
- yaml

## Spuštění
python solveGraphColor.py

## Vstup
Sudoku, nad kterým chcete algoritmus spustit je uloženo v sudoku.yaml. Jméno vstupního souboru je možno změnit pomocí FILE_PATH. 

## Výstup
Nejlepší získané sudoku je vypsáno na stdout. Také je vypsán počet konfliktů a ohodnocení fitness.

## Parametry
Parametry je možné upravit v solveGraphColor.py
- velikost populace: POPULATION_SIZE
- šance křížení: P_CROSSOVER
- šance provedení mutace na jednotlivci: P_MUTATION
    - šance prohození s náhodnou nodou: P_M_SWITCH
    - šance prohození s konfliktní nodou: P_M_CONFLICT_SWITCH
    - šance posunu bloku s nodou: P_M_SHIFT
- počet běhů: RUNS
- generace: MAX_GENERATIONS
- počet nejlepších jedinců z předchozí generace: HALL_OF_FAME_SIZE
- seed: RANDOM_SEED

## Autor
Jaromír Franěk (xfrane16)

## Převzatý kód
https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/tree/master/Chapter05