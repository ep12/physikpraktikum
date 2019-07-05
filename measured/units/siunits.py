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


units_and_prefixes = {k: v for k, v in locals().copy().items() if '__' not in k and isinstance(v, (Unit, UnitComposition, float))}


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
