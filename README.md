`stage=alpha`

# physikpraktikum
tools to simplify physical data analysis

# installation
installation via pip not possible atm
```bash
cd ~/.local/lib/python3.7/site-packages
git clone https://github.com/ep12/physikpraktikum
```

# example
```py
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt

import physikpraktikum as pp


def f(x, a, b, c):
    return a * np.exp(-b * x) + c


xdata = np.array(range(50), dtype=float)
real_params = (23, 17e-2, 0)
ydata = f(xdata, *real_params)
ynoise = np.random.normal(size=xdata.size)
yerr = max(ynoise) * 0.68

plt.style.use(pp.SCISTYLE)

data = {
    'Test': {
        'x': xdata,
        'y': ydata + ynoise,
        'yerr': yerr,
        'label': 'Generated data',
        'fit': {
            'f': f,
            'par': (11, 5e-3, 1),
            'bounds': (-np.inf, np.inf),
            'contx': np.linspace(0, 50, 100),
            'label': 'Fit: $a=${p[0]:.1f}, $b=${p[1]:.2e}, $c=${p[2]:.2f}\n$R^2=${R:.4f}'
        }
    }
}

fig, ax = pp.fit_and_plot(data,
                          title='Fitting Generated Data',
                          xlabel='Time $t$ in seconds',
                          ylabel='Need for QTIplot in arbitrary units',
                          xlim=(-2, 50),
                          ylim=(-2, 25))

fit = data['Test']['fit']
print(fit.keys())
print('Parameters:')
pprint(fit['par'])
print('R^2', fit['R^2'])
print('uncertainty: (abs, rel)')
pprint(fit['par_uncertainty'])
pprint(100*fit['par_uncertainty']/fit['par'])
print('CovarMat:')
pprint(fit['covariance_matrix'])
plt.show()
```
