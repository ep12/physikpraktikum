import numpy as np
import matplotlib.pyplot as plt

import physikpraktikum as pp


def f(x, a, b):
    return a * x + b


xdata = np.array(range(50), dtype=float)
real_params_1 = (2.3, 1200)
real_params_2 = (0.4, 1400)
ydata1 = f(xdata, *real_params_1)
ydata2 = f(xdata, *real_params_2)
ynoise = 25 * np.random.normal(size=xdata.size)
yerr = max(ynoise) * 0.68

fig, (axl, axr) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

axl.plot(xdata, ydata1 + ynoise, label='We')
axl.plot(xdata, ydata2 + ynoise, label='They')
axl.set(title='Management Expert')
axl.legend()

plt.style.use(pp.SCISTYLE)
data = {
    'We': {
        'y': ydata1 + ynoise,
        'x': xdata,
        'fit': {
            'f': f,
            'par': (2, 1000),
            'bounds': (-np.inf, np.inf),
            'label': 'Linear Regression:\n$y={p[0]:.2f}x+{p[1]:.1f}$, $R^2={R:.4f}$',
            'contx': np.linspace(min(xdata), max(xdata) + 1, 100),
        }
    },
    'They': {
        'y': ydata2 + ynoise,
        'x': xdata,
        'fit': {
            'f': f,
            'par': (1, 1000),
            'bounds': (-np.inf, np.inf),
            'label': 'Linear Regression:\n$y={p[0]:.2f}x+{p[1]:.1f}$, $R^2={R:.4f}$',
            'contx': np.linspace(min(xdata), max(xdata) + 1, 100),
        }
    },
}
fig, axr = pp.fit_and_plot(data,
                           title='Physicist',
                           xlabel='Time $t$ in days',
                           ylabel='Income',
                           axes=axr,
                           xlim = (0, max(xdata) + 1),
                           ylim = (1000, 1460))
axr.legend(loc='lower right', fontsize=9, labelspacing=0.5)
axr.grid(True, which='both') # the scistyle is not fully applied :(
axr.grid(which='minor', alpha=0.5)
axr.minorticks_on()
fig.add_subplot(axr)
#plt.show()
plt.savefig('logo.png')
