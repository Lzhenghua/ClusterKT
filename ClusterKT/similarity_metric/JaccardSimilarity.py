def jaccard_similarity(A, B):
    nominator = A.intersection(B)
    denominator = A.union(B)
    similarity = len(nominator)/len(denominator)
    return similarity
similarity = jaccard_similarity(A, B)
print(similarity)
