def mat(matrix, name):
    if name:
        print(f"{name} :")
    print("[")
    for row in matrix:
        print(" ".join(f"{val:10.4f}" for val in row))
    print("]")