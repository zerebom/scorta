def apk(actual: int, predicted: list[int], k: int = 10) -> float:
    if actual in predicted[:k]:
        return 1.0 / (predicted[:k].index(actual) + 1)
    return 0.0


def mapk(actual: list[int], predicted: list[list[int]], k: int = 10):
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)
