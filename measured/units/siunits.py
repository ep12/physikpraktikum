import numpy as np
from physikpraktikum.measured.units.unit import Unit, UnitComposition, UnitPrefix, UnitSystem
#from physikpraktikum.measured.units.unit import Unit, UnitComposition, named_unit_composition as compose, find_single_combined_unit

SI = UnitSystem('SI International System of Units')

Meter = SI('Length', 'Meter', 'm', r'\meter')
Kilogram = SI('Mass', 'Kilogram', 'kg', r'\kilo\gram')
Second = SI('Time', 'Second', 's', r'\second')
Ampere = SI('Electric current', 'Ampere', 'A', r'\ampere')
Kelvin = SI('Thermodynamic temperature', 'Kelvin', 'K', r'\kelvin')
Mole = SI('Amount of substance', 'Mole', 'mol', r'\mole')
Candela = SI('Luminous intensity', 'Candela', 'cd', 'r\candela')

############################################################################################

Radian = SI('Planar angle', 'Radian', 'rad', r'\radian', Meter / Meter) # TODO: siunitx support?
Steradian = SI('Solid angle', 'Steradian', 'sr', r'\steradian', Meter ** 2 / Meter ** 2) # TODO: siunitx support?

Hertz = SI('Frequency', 'Hertz', 'Hz', r'\hertz',  1 / Second)
Becquerel = SI('Radioactivity', 'Becquerel', 'Bq', r'\hertz', 1 / Second) # TODO: siunitx support?
Coulomb = SI('Electric charge', 'Coulomb', 'C', r'\coulomb', Ampere * Second)
Newton = SI('Force', 'Newton', 'N', r'\newton', Kilogram * Meter / Second ** 2)

Pascal = SI('Pressure', 'Pascal', 'Pa', r'\pascal', Newton / Meter ** 2) # TODO: siunitx support?
Joule = SI('Energy', 'Joule', 'J', r'\joule', Newton * Meter) # TODO: siunitx?
Watt = SI('Power', 'Watt', 'W', r'\watt', Joule / Second) # TODO: siunitx?
Volt = SI('Voltage', 'Volt', 'V', r'\volt', Joule / Coulomb)
Farad = SI('Electric capacitance', 'Farad', 'F', r'\farad', Coulomb / Volt)
Ohm = SI('Electric resistance', 'Ohm', 'Ω', r'\ohm', Volt / Ampere)
Siemens = SI('Eletric conductance', 'Siemens', 'S', r'\siemens', 1 / Ohm) # TODO: siunitx?
Weber = SI('Magnetic flux', 'Weber', 'Wb', r'\weber', Volt * Second) # TODO: siunitx?
Tesla = SI('Magnetic induction', 'Tesla', 'T', r'\tesla', Weber / Meter ** 2) # TODO: siunitx?
Henry = SI('Electric inductance', 'Henry', 'H', r'\henry', Weber / Ampere) # TODO: siunitx?
Gray = SI('Absorbed dose of ionizing radiation', 'Gray', 'Gy', r'\gray', Joule / Kilogram) # TODO: siunitx?!
Sievert = SI('Health effect of ionizing radiation', 'Sievert', 'Sv', r'\sievert', Joule / Kilogram) # TODO: siunitx?
Katal = SI('Catalytic activity', 'Katal', 'cat', r'\katal', Mole / Second) # TODO: siunitx?

#all_well_known_units = [v for k, v in locals().copy().items() if '__' not in k and isinstance(v, (Unit, UnitComposition))]


#for u in all_well_known_units:
#    u._kwargs['hooks.print.str'] = print_smart
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

Metre = m = Meter
kg = Kilogram
s = Second
A = Ampere
K = Kelvin
mol = Mole
cd = Candela

############################################################################################


# TODO
# (number * prefix) * unit!
# (10 * milli) * litre!

Minute = SI.add_poor_unit('Minute', 'min', r'\minute', Second, lambda T: 60 * T, lambda t: t / 60)
Hour = SI.add_poor_unit('Hour', 'h', r'\hour', Second, lambda T: 3600 * T, lambda t: t / 3600)
Day = SI.add_poor_unit('Day', 'day', r'\day', Second, lambda T: 86400 * T, lambda t: t / 86400)
JulianYear = SI.add_poor_unit('Julian year', 'year', r'\year', Second, lambda T: 365.25 * 86400 * T, lambda t: t / (365.25 * 86400))


Celsius = SI.add_poor_unit('Degree Celsius', '°C', r'\celsius', Kelvin, lambda c: c + 273.15, lambda k: k - 273.15)
Fahrenheit = SI.add_poor_unit('Degree Fahrenheit', '°F', r'\fahrenheit', Kelvin,
                              lambda x: (x + 459.67) * 5 / 9, lambda x: 9 / 5 * x - 459.67)
Rankine = SI.add_poor_unit('Degree Rankine', '°R', r'\rankine', Kelvin, lambda x: x * 5 / 9, lambda x: x * 9 / 5)
Delisle = SI.add_poor_unit('Degree Delisle', '°De', r'\delisle', Kelvin,
                           lambda x: 373.15 - 2 / 3 * x, lambda x: (373.15 - x) * 3 / 2)
DegreeNewton = SI.add_poor_unit('Degree Newton', '°N', r'\degnewton', Kelvin,
                                lambda x: x * 100 / 33 + 273.15, lambda x: (x - 273.15) * 33 / 100)
Reamur = SI.add_poor_unit('Degree Réaumur', '°Ré', r'\reamur', Kelvin,
                          lambda x: x * 5 / 4 + 273.15, lambda x: (x - 273.15) * 4 / 5)
Romer = SI.add_poor_unit('Degree Rømer', '°Rø', r'\romer', Kelvin,
                         lambda x: (x - 7.5) * 40 / 21 + 273.15, lambda x: (x - 273.15) * 21 / 40 + 7.5) # THE WORST!


ImperialInch = SI.add_poor_unit('Imperial Inch', 'in', r'\inch', Meter, lambda M: 25.4e-3 * M, lambda k: k / 25.4e-3)
ImperialFoot = SI.add_poor_unit('Imperial Foot', 'ft', r'\foot', Meter, lambda M: 0.3048 * M, lambda k: k / 0.3048)
ImperialYard = SI.add_poor_unit('Imperial Yard', 'yd', r'\yard', Meter, lambda M: 0.9144 * M, lambda k: k / 0.9144)
ImperialMile = SI.add_poor_unit('Imperial Mile', 'mi', r'\mile', Meter, lambda M: 1609.344 * M, lambda k: k / 1609.344)
NauticalMile = SI.add_poor_unit('Nautical Mile', 'nmi', r'\nauticalmile', Meter, lambda M: 1852 * M, lambda k: k / 1852)
Smoot = SI.add_poor_unit('Smoot', 'smoot', r'\smoot', Meter, lambda S: S * 1.702, lambda k: k / 1.702)
AstronomicalUnit = SI.add_poor_unit('Astronomical unit', 'au', r'\astronomicalunit', Meter,
                                    lambda B: 149597870700 * B, lambda m: m / 149597870700) # defined: 149597870700 m
Lightyear = SI.add_poor_unit('Lightyear', 'ly', r'\lightyear', Meter,
                             lambda M: (365.25 * 86400 * 299792358) * M, lambda k: k / (365.25 * 86400 * 299792358)) # 1 year * c
Parsec = SI.add_poor_unit('Parsec', 'pc', r'\parsec', Meter,
                          lambda P: (648000 / np.pi * 149597870700) * P, lambda k: k / (648000 / np.pi * 149597870700)) # 648000/pi AU


Hectare = SI.add_poor_unit('Hectare', 'ha', r'\hectare', Meter ** 2,
                           lambda A: 1e4 * A, lambda H: H * 1e-4)


Litre = SI.add_poor_unit('Litre', 'L', r'\litre', Meter ** 3,
                         lambda L: L * 1e-3, lambda k: k * 1e3)


MetricTonne = SI.add_poor_unit('Metric tonne', 't', r'\tonne', Kilogram, lambda x: 1e3 * x, lambda x: 1e-3 * x)
Gram = SI.add_poor_unit('Gram', 'g', r'\gram', Kilogram, lambda x: 1e-3 * x, lambda x: 1e3 * x)
AtomicMassUnit = SI.add_poor_unit('Atomic mass unit', 'AMU', r'\amu', Kilogram, lambda x: 1.66053906660e-27 * x, lambda x: x / 1.66053906660e-27)
Dalton = AMU = AtomicMassUnit


Electronvolt = SI.add_poor_unit('Electronvolt', 'eV', r'\electronvol', Joule, lambda x: 1.602176634e-19 * x, lambda x: x / 1.602176634e-19)


Neper = SI.add_poor_unit('Neper', 'Np', r'\neper', 1, lambda x: np.exp(x), lambda x: np.log(x))


if __name__ == '__main__':
    test = [
        #Meter / Meter, # 1
        #1 / Second, # Hz
        #Ampere * Second, # C
        #Kilogram * Meter * Hertz ** 2, # N
        #Newton / Meter ** 2, # Pa
        #Newton * Meter, # J
        #Joule / Second, # W
        #Watt * Second, # J
        #Joule / Coulomb, # V
        #Farad * Volt, # C <-
        Ampere / Sievert, # <-
        #Tesla * Meter ** 2, # Wb
        #Katal * Second, # Mole
        #Gray * Kilogram, #  J
        #Ampere * Henry # Wb
    ]
    for x in test:
        print(x, ':', x.as_base_units)
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
