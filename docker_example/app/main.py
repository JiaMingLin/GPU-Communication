import numpy as np

def main():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = A @ B
    print("Matrix multiplication result:\n", C)

if __name__ == "__main__":
    main()