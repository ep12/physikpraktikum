import numpy as np
from physikpraktikum.measured.units.unit import Unit, UnitComposition, UnitPrefix, UnitSystem
from physikpraktikum.utils.representations import longstr, describe

SI = UnitSystem('SI International System of Units', limit_combined_units = 4)

Meter = SI('Length', 'Meter', 'm', r'\meter')
Kilogram = SI('Mass', 'Kilogram', 'kg', r'\kilo\gram')
Second = SI('Time', 'Second', 's', r'\second')
Ampere = SI('Electric current', 'Ampere', 'A', r'\ampere')
Kelvin = SI('Thermodynamic temperature', 'Kelvin', 'K', r'\kelvin')
Mole = SI('Amount of substance', 'Mole', 'mol', r'\mole')
Candela = SI('Luminous intensity', 'Candela', 'cd', r'\candela')

############################################################################################

Radian = SI('Planar angle', 'Radian', 'rad', r'\radian', Meter / Meter)
Steradian = SI('Solid angle', 'Steradian', 'sr', r'\steradian', Meter ** 2 / Meter ** 2)

Hertz = SI('Frequency', 'Hertz', 'Hz', r'\hertz',  1 / Second)
Becquerel = SI('Radioactivity', 'Becquerel', 'Bq', r'\becquerel', 1 / Second)
Coulomb = SI('Electric charge', 'Coulomb', 'C', r'\coulomb', Ampere * Second)
Newton = SI('Force', 'Newton', 'N', r'\newton', Kilogram * Meter / Second ** 2)

Pascal = SI('Pressure', 'Pascal', 'Pa', r'\pascal', Newton / Meter ** 2)
Joule = SI('Energy', 'Joule', 'J', r'\joule', Newton * Meter)
Watt = SI('Power', 'Watt', 'W', r'\watt', Joule / Second)
Volt = SI('Voltage', 'Volt', 'V', r'\volt', Joule / Coulomb)
Farad = SI('Electric capacitance', 'Farad', 'F', r'\farad', Coulomb / Volt)
Ohm = SI('Electric resistance', 'Ohm', 'Ω', r'\ohm', Volt / Ampere)
Siemens = SI('Eletric conductance', 'Siemens', 'S', r'\siemens', 1 / Ohm)
Weber = SI('Magnetic flux', 'Weber', 'Wb', r'\weber', Volt * Second)
Tesla = SI('Magnetic induction', 'Tesla', 'T', r'\tesla', Weber / Meter ** 2)
Henry = SI('Electric inductance', 'Henry', 'H', r'\henry', Weber / Ampere)
Gray = SI('Absorbed dose of ionizing radiation', 'Gray', 'Gy', r'\gray', Joule / Kilogram)
Sievert = SI('Health effect of ionizing radiation', 'Sievert', 'Sv', r'\sievert', Joule / Kilogram)
Katal = SI('Catalytic activity', 'Katal', 'cat', r'\katal', Mole / Second)

############################################################################################

yocto = SI.add_prefix(UnitPrefix('Yocto', 'y', r'\yocto', 1e-24)) # TODO: siunitx?
zepto = SI.add_prefix(UnitPrefix('Zepto', 'z', r'\zepto', 1e-21))
atto = SI.add_prefix(UnitPrefix('Atto', 'a', r'\atto', 1e-18))
femto = SI.add_prefix(UnitPrefix('Femto', 'f', r'\femto', 1e-15))
pico = SI.add_prefix(UnitPrefix('Pico', 'p', r'\pico', 1e-12))
nano = SI.add_prefix(UnitPrefix('Nano', 'n', r'\nano', 1e-9))
micro = SI.add_prefix(UnitPrefix('Micro', 'µ', r'\micro', 1e-6))
milli = SI.add_prefix(UnitPrefix('Milli', 'm', r'\milli', 1e-3))
centi = SI.add_prefix(UnitPrefix('Centi', 'c', r'\centi', 1e-2))
dezi = SI.add_prefix(UnitPrefix('Deci', 'd', r'\deci', 1e-1))
Deka = SI.add_prefix(UnitPrefix('Deca', 'da', r'\deca', 1e1))
Hecto = SI.add_prefix(UnitPrefix('Hecto', 'h', r'\hecto', 1e2))
Kilo = SI.add_prefix(UnitPrefix('Kilo', 'k', r'\kilo', 1e3))
Mega = SI.add_prefix(UnitPrefix('Mega', 'M', r'\mega', 1e6))
Giga = SI.add_prefix(UnitPrefix('Giga', 'G', r'\giga', 1e9))
Tera = SI.add_prefix(UnitPrefix('Tera', 'T', r'\tera', 1e12))
Peta = SI.add_prefix(UnitPrefix('Peta', 'P', r'\peta', 1e15))
Exa = SI.add_prefix(UnitPrefix('Exa', 'E', r'\exa', 1e18))
Zetta = SI.add_prefix(UnitPrefix('Zetta', 'Z', r'\zetta', 1e21))
Yotta = SI.add_prefix(UnitPrefix('Yotta', 'Y', r'\yotta', 1e24))

############################################################################################

# This list is probably not complete!
SI._definitely_as_base_units.append(Meter / Second) # would be sqrt(Sv) which is obviously shorter but not intuitive
SI._definitely_as_base_units.append(Meter / Second ** 2) # would be sqrt(Sv) * Bq

############################################################################################


# TODO
# (number * prefix) * unit!
# (10 * milli) * litre! vs 10 * (milli * litre)

Degree = SI.add_poor_unit('Degree', '°', r'\degree', Radian,
                          lambda k: k * np.pi / 180, lambda k: k * 180 / np.pi, is_linear=True)
Arcminute = SI.add_poor_unit('Arcminute', 'arcmin', r'\arcminute', Radian,
                             lambda k: k * np.pi / 10800, lambda k: k * 10800 / np.pi, is_linear=True)
Arcsecond = SI.add_poor_unit('Arcsecond', 'arcsec', r'\arcsecond', Radian,
                             lambda k: k * np.pi / 648000, lambda k: k * 648000 / np.pi, is_linear=True)
Gon = SI.add_poor_unit('Gon', 'gon', r'\gradian', Radian, # TODO: siunitx: no gradian!
                       lambda k: k * np.pi / 200, lambda k: k * 200 / np.pi, is_linear=True)
Gradian = Gon


Minute = SI.add_poor_unit('Minute', 'min', r'\minute', Second, lambda T: 60 * T, lambda t: t / 60, is_linear=True)
Hour = SI.add_poor_unit('Hour', 'h', r'\hour', Second, lambda T: 3600 * T, lambda t: t / 3600, is_linear=True)
Day = SI.add_poor_unit('Day', 'day', r'\day', Second, lambda T: 86400 * T, lambda t: t / 86400, is_linear=True)
JulianYear = SI.add_poor_unit('Julian year', 'year', r'\year', Second, # TODO: siunitx: no year!
                              lambda T: 365.25 * 86400 * T, lambda t: t / (365.25 * 86400), is_linear=True)


Celsius = SI.add_poor_unit('Degree Celsius', '°C', r'\celsius', Kelvin, lambda c: c + 273.15, lambda k: k - 273.15) # siunitx: degreeCelsius?
Fahrenheit = SI.add_poor_unit('Degree Fahrenheit', '°F', r'\fahrenheit', Kelvin, # siunitx: ?
                              lambda x: (x + 459.67) * 5 / 9, lambda x: 9 / 5 * x - 459.67)
Rankine = SI.add_poor_unit('Degree Rankine', '°R', r'\rankine', Kelvin, lambda x: x * 5 / 9, lambda x: x * 9 / 5) # siunitx: ?
Delisle = SI.add_poor_unit('Degree Delisle', '°De', r'\delisle', Kelvin,
                           lambda x: 373.15 - 2 / 3 * x, lambda x: (373.15 - x) * 3 / 2) # siunitx: ?
DegreeNewton = SI.add_poor_unit('Degree Newton', '°N', r'\degnewton', Kelvin,
                                lambda x: x * 100 / 33 + 273.15, lambda x: (x - 273.15) * 33 / 100) # siunitx: ?
Reamur = SI.add_poor_unit('Degree Réaumur', '°Ré', r'\reamur', Kelvin,
                          lambda x: x * 5 / 4 + 273.15, lambda x: (x - 273.15) * 4 / 5) # siunitx: ?
Romer = SI.add_poor_unit('Degree Rømer', '°Rø', r'\romer', Kelvin,
                         lambda x: (x - 7.5) * 40 / 21 + 273.15, lambda x: (x - 273.15) * 21 / 40 + 7.5) # siunitx: ? # THE WORST!


ImperialInch = SI.add_poor_unit('Imperial Inch', 'in', r'\inch', Meter,
                                lambda M: 25.4e-3 * M, lambda k: k / 25.4e-3, is_linear=True) # siunitx: ?
ImperialFoot = SI.add_poor_unit('Imperial Foot', 'ft', r'\foot', Meter,
                                lambda M: 0.3048 * M, lambda k: k / 0.3048, is_linear=True) # siunitx: ?
ImperialYard = SI.add_poor_unit('Imperial Yard', 'yd', r'\yard', Meter,
                                lambda M: 0.9144 * M, lambda k: k / 0.9144, is_linear=True) # siunitx: ?
ImperialMile = SI.add_poor_unit('Imperial Mile', 'mi', r'\mile', Meter,
                                lambda M: 1609.344 * M, lambda k: k / 1609.344, is_linear=True) # siunitx: ?
NauticalMile = SI.add_poor_unit('Nautical Mile', 'nmi', r'\nauticalmile', Meter,
                                lambda M: 1852 * M, lambda k: k / 1852, is_linear=True)
Smoot = SI.add_poor_unit('Smoot', 'smoot', r'\smoot', Meter, lambda S: S * 1.702, lambda k: k / 1.702, is_linear=True) # siunitx: ?
AstronomicalUnit = SI.add_poor_unit('Astronomical unit', 'AU', r'\astronomicalunit', Meter,
                                    lambda B: 149597870700 * B, lambda m: m / 149597870700, is_linear=True) # defined: 149597870700 m
Lightyear = SI.add_poor_unit('Lightyear', 'ly', r'\lightyear', Meter,
                             lambda M: (365.25 * 86400 * 299792358) * M, lambda k: k / (365.25 * 86400 * 299792358), # siunitx: ?
                             is_linear=True) # 1 year * c
Parsec = SI.add_poor_unit('Parsec', 'pc', r'\parsec', Meter,
                          lambda P: (648000 / np.pi * 149597870700) * P, lambda k: k / (648000 / np.pi * 149597870700), # siunitx: ?
                          is_linear=True) # 648000/pi AU


Hectare = SI.add_poor_unit('Hectare', 'ha', r'\hectare', Meter ** 2,
                           lambda A: 1e4 * A, lambda H: H * 1e-4)


Litre = SI.add_poor_unit('Litre', 'L', r'\litre', Meter ** 3,
                         lambda L: L * 1e-3, lambda k: k * 1e3)


MetricTonne = SI.add_poor_unit('Metric tonne', 't', r'\tonne', Kilogram, lambda x: 1e3 * x, lambda x: 1e-3 * x, is_linear=True)
Gram = SI.add_poor_unit('Gram', 'g', r'\gram', Kilogram, lambda x: 1e-3 * x, lambda x: 1e3 * x, is_linear=True)
AtomicMassUnit = SI.add_poor_unit('Atomic mass unit', 'AMU', r'\atomicmassunit', Kilogram,
                                  lambda x: 1.66053906660e-27 * x, lambda x: x / 1.66053906660e-27, is_linear=True)
Dalton = AMU = AtomicMassUnit


Electronvolt = SI.add_poor_unit('Electronvolt', 'eV', r'\electronvolt', Joule,
                                lambda x: 1.602176634e-19 * x, lambda x: x / 1.602176634e-19, is_linear=True)


Neper = SI.add_poor_unit('Neper', 'Np', r'\neper', 1, lambda x: np.exp(x), lambda x: np.log(x))


if __name__ == '__main__':
    from pprint import pprint
    print(SI)
    print(2 * Minute)
    print(30 / Minute)
    print(60 / Minute)
    print(120 / Minute)
    print(37 * Celsius)
    print(100 * Celsius)
    print(1 * AstronomicalUnit)
    print(1 * Lightyear)
    print(0 * Romer)
    print(20e9 * Electronvolt)
    print(0 * Neper)
    print(17 * Newton)
    print(2 * Meter * 4 * Meter)
    print((2 * Meter * 4 * Meter) ** (-1))
    print(17 * Newton / (2 * Meter * 4 * Meter))
    Area = ((Meter * np.array([1, 2, 3])) @ (Meter * np.array([-1, 0, 1])))
    Force = 50 * Newton
    print(Force / Area)
    vec1 = np.array([1, 2, 3]) * Meter
    #print(vec1 @ np.array([-1, 0, 1]))
    #pprint(vec1.__dir__())
    #print(Area, type(Area))
    #pprint(Area.__dict__)
    print(([1, 2, 3] * Meter) @ ([7, 8, 9] * Hertz))
    print()
    x = ([3, 4] * Meter) @ [2, 5]
    print(2 * x)
    x = 7 * Second * 3 * Hertz
    print(x, type(x))
    #for a in dir(x):
    #    print(a, '\t', getattr(x, a))
    #print(SI.as_vector(Weber))
    #print(SI.base_units.keys())
    print(Meter ** 2 * Kilogram * Second ** (-2) * Ampere ** (-1))
    print(Newton * Candela)
    #pprint(SI.as_matrix, width=240)
    print([1, 2, 3] * Meter * 5 * Meter)
    v = [80, 43, 21] * Kilo * Meter / (1 * Hour)
    print(v)
    print(v.norm)
    print([80, 0, 0] * Kilo)
    print((Joule / Kilogram).as_base_units)
    print(Meter / Second)
    print(Meter / Second ** 2)
    print(np.sin(5 * Hertz * 10 * Second) * 10 * Volt)
    x = Newton * Kelvin
    print(x, type(x))
    y = SI._optise_vector_combination(x)
    E = [7, 1, -2] * Volt / Meter
    mu_0 = 1.2566370614359173e-6 * Newton / Ampere ** 2
    B = [4, -2, 1] * micro * Tesla
    H = B / mu_0
    print(E.cross(H)) # Correctly calculates the poynting vector with the unit Watt per square meter
    print((Watt / Meter ** 2).as_base_units)
    x = Watt * Meter / Kelvin * Candela ** 2
    print(x)
    print(x.as_base_units(), '=', x)
    print(Meter ** 2 * Kilogram * Second * Ampere * Kelvin / Candela / Mole)
    print(Farad * Newton * Ampere ** (-2) / Meter * Joule * Second)
    #  C F kg Hz J^3 N A^-2 K^-2 mol^-1 T^-1  # Unitful.jl outputs 10 cluttered units
    x = Coulomb * Farad * Kilogram * Hertz * Joule ** 3 * Newton / (Ampere ** 2 * Kelvin ** 2 * Mole * Tesla)
    print(x.as_base_units, '=', x) # We need 6 == 6 base units are in this product
    print(Ampere * Second / (Volt * Meter))

    vol = 1000 * (milli * Litre)
    #print((1000 * milli) * Litre)
    print('1 L', 1 * Litre)
    print('1000 mL', 1000 * (milli * Litre), (1000 * milli) * Litre)
    print('1 mL', milli * Litre)
    print('1 in', 1 * ImperialInch)
    print('1 in^2', (1 * ImperialInch) ** 2)
    print('10 in^2', 10 * ImperialInch ** 2)
    print('10 ft^3', 10 * ImperialFoot ** 3)
    print(Litre.from_base(1000 * (milli * Litre)))
    print('10 ha', 10 * Hectare)
    print('10 mha', 10 * milli * Hectare)
    print('10 (cm)^3', 10 * (centi * Meter) ** 3)
    print('10 c m^3', 10 * centi * Meter ** 3)
    print(r'\documentclass{article}')
    print(r'\usepackage{siunitx}')
    print(r'\begin{document}')
    for k, v in SI.units.items():
        print(k, '(%s)' % v.tex.replace('\\', '+'), r': $\SI{1.0}{', v.tex, r'}$\\', sep='')
    print(r'\end{document}')
    print(longstr(Meter))
    print(longstr(Newton), longstr(Newton ** 2 * Meter ** (11)), longstr(Newton ** (17) * Candela ** 2))
    test = ['Meter', 'Kilogram', 'kg', 'cd', 'T', 'g', 'in', 'milli', 't', 'Nanofarad', 'Sv K⁻¹', '(Sv K)⁻¹',
            'cm', 'smoot', 'AU', 'mT', 'µH', 'J g^-1 K^-1', 'J (g K)^-1', 'J (g (cm kT)^3 K)^-2']
    #test = ['(Sv K)⁻¹']
    for x in test:
        y = 2 * SI.find_unit_from_string(x)
        print(x, type(y), y)
