from physikpraktikum.measured.units.unit import Unit, UnitComposition, named_unit_composition as compose, find_single_combined_unit

Meter = Unit('Length', 'Meter', 'm', r'\meter')
Kilogram = Unit('Mass', 'Kilogram', 'kg', r'\kilo\gram')
Second = Unit('Time', 'Second', 's', r'\second')
Ampere = Unit('Electric current', 'Ampere', 'A', r'\ampere')
Kelvin = Unit('Thermodynamic temperature', 'Kelvin', 'K', r'\kelvin')
K = Kelvin
Mole = Unit('Amount of substance', 'Mole', 'mol', r'\mole')
Candela = Unit('Luminous intensity', 'Candela', 'cd', 'r\candela')

############################################################################################

Hertz = compose(1 / Second, 'Frequency', 'Hertz', 'Hz', r'\hertz')
Becquerel = compose(1 / Second, 'Radioactivity', 'decays per second', 'Bq', r'\hertz') # TODO: siunitx support?
Coulomb = compose(Ampere * Second, 'Electric charge', 'Coulomb', 'C', r'\coulomb')
Newton = compose(Kilogram * Meter / Second ** 2, 'Force', 'Newton', 'N', r'\newton')

Pascal = compose(Newton / Meter ** 2, 'Pressure', 'Pascal', 'Pa', r'\pascal') # TODO: siunitx support?
Joule = compose(Newton * Meter, 'Energy', 'Joule', 'J', r'\joule') # TODO: siunitx?
Watt = compose(Joule / Second, 'Power', 'Watt', 'W', r'\watt') # TODO: siunitx?
Volt = compose(Joule / Coulomb, 'Voltage', 'Volt', 'V', r'\volt')
Farad = compose(Coulomb / Volt, 'Electric capacitance', 'Farad', 'F', r'\farad')
Ohm = compose(Volt / Ampere, 'Electric resistance', 'Ohm', 'Ω', r'\ohm')
Siemens = compose(1 / Ohm, 'Eletric conductance', 'Siemens', 'S', r'\siemens') # TODO: siunitx?
Weber = compose(Volt * Second, 'Magnetic flux', 'Weber', 'Wb', r'\weber') # TODO: siunitx?
Tesla = compose(Weber / Meter ** 2, 'Magnetic induction', 'Tesla', 'T', r'\tesla') # TODO: siunitx?
Henry = compose(Weber / Ampere, 'Electric inductance', 'Henry', 'H', r'\henry') # TODO: siunitx?
Gray = compose(Joule / Kilogram, 'Absorbed dose of ionizing radiation', 'Gray', 'Gy', r'\gray') # TODO: siunitx?!
Sievert = compose(Joule / Kilogram, 'Health effect of ionizing radiation', 'Sievert', 'Sv', r'\sievert') # TODO: siunitx?
Katal = compose(Mole / Second, 'Catalytic activity', 'Katal', 'cat', r'\katal') # TODO: siunitx?

all_well_known_units = [v for k, v in locals().copy().items() if '__' not in k and isinstance(v, (Unit, UnitComposition))]


#for u in all_well_known_units:
#    u._kwargs['hooks.print.str'] = print_smart
############################################################################################

yocto = 10e-24
zepto = 10e-21
atto = 10e-18
femto = 10e-15
pico = 10e-12
nano = 10e-9
micro = 10e-6
milli = 10e-3
centi = 10e-2
dezi = 10e-1
Deka = 10e1
Hecto = 10e2
Kilo = 10e3
Mega = 10e6
Giga = 10e9
Tera = 10e12
Peta = 10e15
Exa = 10e18
Zetta = 10e21
Yotta = 10e24

############################################################################################

Metre = m = Meter
kg = Kilogram
s = Second
A = Ampere
K = Kelvin
mol = Mole
cd = Candela


units_and_prefixes = {k: v for k, v in locals().copy().items() if '__' not in k and isinstance(v, (Unit, UnitComposition, float))}


if __name__ == '__main__':
    #for x in [Second, Kilogram, Meter, Ampere, Candela, Kelvin, Mole]:
    #    print('%r' % x)
    #print(Hertz)
    #print(Newton)
    #print(Newton * Second ** 2 / Meter)
    #for x in all_well_known_units:
    #    print(x, repr(x))
    #print(Joule / (Kilogram * Kelvin))
    #print(Weber / Meter ** 2 == Tesla)
    print('HERE' *  10)
    #test = Joule / Weber
    test = Hertz / Newton
    print(test._kwargs)
    #for k, v in test._kwargs.items():
    #    print(k, v)
