import matplotlib.pyplot as plt
import stohastic_signals_filtration as ssf
import numpy as np


def plot_results(x_source, y_source,
                 y_noisy, y_filtered,
                 title, filename):
    plt.title(title)
    plt.plot(x_source, y_source, color='green')
    plt.plot(x_source, y_noisy, color='red')
    plt.plot(x_source, y_filtered, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(linestyle='--')
    plt.legend(['Normal', 'Noisy', 'Filtered'])
    plt.savefig(filename)
    plt.clf()


def plot_criteria(noise, difference,
                  filename):
    plt.xlabel('Noise criteria')
    plt.xscale('log')
    plt.ylabel('Difference criteria')
    plt.yscale('log')
    colors = np.random.rand(len(noise))
    plt.scatter(noise, difference, c=colors)
    plt.grid(linestyle='--')
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    # Functions to work on
    function = lambda x: np.sin(x) + 0.5
    noisy_function = ssf.uniform_noise(function, amp=0.5)

    # Source data
    arrangement = (0., 1., 11)
    tries = ssf.get_tries(x_min=0.,
                          x_max=np.pi,
                          eps=0.01,
                          proba=0.95)
    x_source = np.linspace(start=0.,
                           stop=np.pi,
                           num=101)
    y_source = function(x_source)
    y_noisy = noisy_function(x_source)

    # r = 3
    i, results = ssf.find_weights(y_noisy, 3, tries, arrangement)
    results.to_csv('results_3.csv')
    weights = results.loc[i]['Alpha']
    y_filtered = ssf.mean_geom_sw(y_noisy, weights)
    plot_results(x_source, y_source, y_noisy, y_filtered, 'sin(x) + 0.5', 'results_3.png')
    plot_criteria(results['W'].values, results['D'].values, 'criteria_3.png')

    # r = 5
    i, results = ssf.find_weights(y_noisy, 5, tries, arrangement)
    results.to_csv('results_5.csv')
    weights = results.loc[i]['Alpha']
    y_filtered = ssf.mean_geom_sw(y_noisy, weights)
    plot_results(x_source, y_source, y_noisy, y_filtered, 'sin(x) + 0.5', 'results_5.png')
    plot_criteria(results['W'].values, results['D'].values, 'criteria_5.png')

    print('Done!')

