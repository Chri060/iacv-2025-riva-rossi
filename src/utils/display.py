# Shows matrices in a nice way
def mat(matrix, name=None):
    if name:
        print(f"{name} :")
    print("[")
    for row in matrix:
        print(" ".join(f"{val:10.4f}" for val in row))
    print("]")

def title_print(title_text):
    print(f"\n\n<[---------------------{title_text}---------------------]>\n\n")