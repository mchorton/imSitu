# This file defines integration tests.
# These are mostly scripts that I shouldn't break, e.g. this just makes sure
# that I don't e.g. delete needed functions, and is useful while developing.
# It doesn't make sure that the code functions as intended!
def main():
    import split.splitters as spsp
    spsp.splitTrainDevTestMinInTrain("testing/_int_spsptest", test=True)

if __name__ == '__main__':
    main()
