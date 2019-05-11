# 2019-05-11
# 하나고등학교 LAMBDA
# Hana Academy Seoul LAMBDA
# https://github.com/haslambda
#
# 영진이의 유전 알고리즘(근사) 강의
# Genetic algorithm for polynomial approximation taught by Youngjin Song
#
# 이윤규 번역
# Arranged by Lucas "rocketll" Yunkyu Lee
#
# Tested on Python 3.7.3

import numpy as np
import matplotlib.pyplot as plt  # 그래프 생성용 / generates plot
import types  # 람다 자료형 / lambda type

generation = []
POPULATION = 200  # 개체 200마리 / 200 individuals
GENES = 7  # 6차 근사 / 6th degree approximation
NEIGHBORHOOD = np.linspace(-2, 2, 2001)
E = 0.1  # 돌연변이 확률 / chance of mutation

FUNC = lambda x: np.sin(x)  # 근사할 함수 sin(x) / function to approximate sin(x)

TOP = 20  # 부모 개체 수 / number of parent individuals

start, end = -5, 5 # 유전자의 난수 범위 / range for gene rng


def create_generation() -> list:
    """
    세대 생성
    Creates generations
    """
    generation = []
    for _ in range(POPULATION):
        individual = []
        for _ in range(POPULATION):
            individual.append(np.random.uniform(start, end))
        generation.append(individual)
    return generation


def mutation(indi: list) -> list:
    """
    랜덤으로 돌연변이 생성
    Creates mutations randomly
    """
    individual = []
    for i in range(GENES):
        if np.random.random() < E:
            individual.append(np.random.uniform(start, end))
        else:
            individual.append(indi[i])
    return individual


def crossover(indi1: list, indi2: list) -> list:
    """
    두 개체를 교배
    Crossovers two individuals
    """
    individual = []
    for i in range(GENES):
        individual.append(np.random.uniform(indi1[i], indi2[i]))
    return individual


def fitness_func(indi: list) -> float:
    """
    sin(x)와 비교하는 적합성 함수
    Fitness function to compare with sin(x) 
    """
    indi_func = (
        lambda x: indi[0]
        + indi[1] * x
        + indi[2] * x ** 2
        + indi[3] * x ** 3
        + indi[4] * x ** 4
        + indi[5] * x ** 5
        + indi[6] * x ** 6
    )
    sample = map(lambda x: indi_func(x), NEIGHBORHOOD)
    std = map(lambda x: FUNC(x), NEIGHBORHOOD)
    lm = sum(map(lambda x, y: (x - y) ** 2, sample, std))
    return lm


def natural_selection(gen: list, top: int = TOP) -> list:
    """
    TOP만큼 자연선택
    Naturally select TOP individuals
    """
    return sorted(gen, key=lambda x: fitness_func(x))[:top]


def group_crossover(group: list) -> list:
    """
    세대 전체 교배
    Crossover for entire generation
    """
    result = []
    for i in group:
        for j in group:
            result.append(crossover(i, j))
    return result


def group_mutation(group: list) -> list:
    """
    세대 전체 돌연변이
    Mutates entire generation
    """
    result = []
    for i in group:
        result.append(mutation(i))
    return result


def next_generation(gen: list) -> list:
    """
    다음 세대 진화
    Evolution into next generation
    """
    good_gene = natural_selection(gen)
    generation = good_gene.copy()
    generation.extend(group_crossover(good_gene)[: POPULATION - TOP])
    generation = group_mutation(generation)
    return generation


def plot(result: types.LambdaType):
    """
    sin(x)와 생성된 다항함수 비교하는 그래프 생성
    Compares sin(x) with the generated polynomial
    """
    true_sin = np.sin(NEIGHBORHOOD)
    approx_sin = np.asarray(list(map(result, NEIGHBORHOOD)))
    plt.plot(NEIGHBORHOOD, true_sin)
    plt.plot(NEIGHBORHOOD, approx_sin)

    plt.show()


def main():
    """
    실행되는 main 함수
    main function to execute
    """
    generation = create_generation()
    for i in range(200):
        print(f"generation {i+1} num : {len(generation)} --------------")
        generation = next_generation(generation)
        print(f"{generation[0]}")

    print(f"{natural_selection(generation, 1)}")

    res = natural_selection(generation, 1)[0]
    print(res)
    result = (
        lambda x: res[0]
        + res[1] * x
        + res[2] * x ** 2
        + res[3] * x ** 3
        + res[4] * x ** 4
        + res[5] * x ** 5
        + res[6] * x ** 6
    )
    plot(result)


if __name__ == "__main__":
    main()
