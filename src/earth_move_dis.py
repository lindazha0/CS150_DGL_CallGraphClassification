# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
from scipy.stats import wasserstein_distance

V1 = [0, 1, 3]
V2 = [5, 6, 8]

dist = wasserstein_distance(V1, V2)
print(f"Wasserstein Distance: {dist}")