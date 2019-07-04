from physikpraktikum.measured.units.unit import Unit, UnitComposition, UnitSystem
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

Hertz = SI('Frequency', 'Hertz', 'Hz', r'\hertz', 1 / Second)
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
    print('HERE ' *  10)
    #test = Joule / Weber
    test = Hertz / Newton
    print(test)
    print(SI)
    #for k, v in test._kwargs.items():
    #    print(k, v)
    print(SI.str_vector_representation)
