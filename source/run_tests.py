from sklearn.datasets import load_wine

from source.test_utils import create_gaussian_5, create_gaussian_25, run_test, save_metrics, initialize_algorithms


def test_gaussian(algorithms, out_file, option=0, verbose=False):
    if option == 0:
        print("Test Gaussian 5")
        data, gs = create_gaussian_5()
    elif option == 1:
        print("Test Gaussian 25")
        data, gs = create_gaussian_25()
    else:
        raise ValueError(f"There is no option {option}. Available options are [0, 1].")

    metrics = run_test(out_file, algorithms, data, gs, verbose=verbose)

    save_metrics(out_file, metrics)


def test_wine(algorithms, out_file, verbose=False):
    wine = load_wine()
    metrics = run_test(algorithms, wine.data, wine.target, verbose=verbose)
    save_metrics(out_file, metrics)


test_gaussian(initialize_algorithms(5, 1e-4, 300), "gaussian_5", option=0, verbose=True)

test_gaussian(initialize_algorithms(25, 1e-4, 300), "gaussiang_25", option=1, verbose=True)

# test_wine('wine', initialize_algorithms(3, 1e-4, 300), "wine", verbose=True)
