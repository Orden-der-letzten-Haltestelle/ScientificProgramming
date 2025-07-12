import unittest
import functions as f
import math

class TestConstFunc(unittest.TestCase):
    """
    Unittests für die konstante Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f  = f.ConstFunc("f", 1.0)
        self.g  = f.ConstFunc("g", 3.5)
        self.h  = f.ConstFunc("h", -2.0, 'y')
        self.k  = f.ConstFunc("k", 0.0)

    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(1.0), 1.0)
        self.assertAlmostEqual(self.f(-10.0), 1.0)
        self.assertAlmostEqual(self.g(0.0), 3.5)
        self.assertAlmostEqual(self.h(0.0), -2.0)
        self.assertAlmostEqual(self.k(11.5), 0.0)

    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),'f(x)=1.0')
        self.assertEqual(str(self.g),'g(x)=3.5')
        self.assertEqual(str(self.h),'h(y)=-2.0')
        self.assertEqual(str(self.k),'k(x)=0.0')

    def test_03_derive(self):
        """
        Testet Ableitung.
        """
        fd= self.f.derive()
        hd= self.h.derive()
        self.assertTrue(type(fd) is f.ConstFunc)
        self.assertEqual(fd.factor, 0.0)
        self.assertEqual(fd.operand, 'x')
        self.assertEqual(str(fd), "f'(x)=0.0")
        self.assertTrue(type(hd) is f.ConstFunc)
        self.assertEqual(hd.factor, 0.0)
        self.assertEqual(hd.operand, 'y')
        self.assertEqual(str(hd), "h'(y)=0.0")


class TestExpFunc(unittest.TestCase):
    """
    Unittests für die e-Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f  = f.ExpFunc("f", 1.0)
        self.g  = f.ExpFunc("g", 2.0, 3.0)
        self.h  = f.ExpFunc("h", -1.0, -2.0, 'y')

    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(0.0), 1.0)
        self.assertAlmostEqual(self.f(1.0), math.e)
        self.assertAlmostEqual(self.f(3.0), 20.085536923187664)
        self.assertAlmostEqual(self.g(2.0), 806.85758698547)
        self.assertAlmostEqual(self.g(-2.0), 0.004957504353332719)
        self.assertAlmostEqual(self.h(4.0), -0.00033546262790251196)
        self.assertAlmostEqual(self.h(-4.0), -2980.957987041727)

    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),'f(x)=e^x')
        self.assertEqual(str(self.g),'g(x)=2.0e^(3.0x)')
        self.assertEqual(str(self.h),'h(y)=-e^(-2.0y)')

    def test_03_derive(self):
        """
        Testet Ableitung.
        """
        fd= self.f.derive()
        gd= self.g.derive()
        hd= self.h.derive()
        self.assertTrue(type(fd) is f.ExpFunc)
        self.assertEqual(fd.name, "f'")
        self.assertEqual(fd.factor, 1.0)
        self.assertEqual(fd.exp_factor, 1.0)
        self.assertEqual(fd.operand, 'x')
        self.assertEqual(str(gd),"g'(x)=6.0e^(3.0x)")
        self.assertAlmostEqual(gd(2.5), 10848.254486736376)
        self.assertEqual(str(hd),"h'(y)=2.0e^(-2.0y)")
        self.assertAlmostEqual(hd(1.0), 0.2706705664732254)

class TestSinCos(unittest.TestCase):
    """
    Unittests für die Sinus- und Cosinus-Funktionen.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.fs  = f.SinFunc("f", 1.0)
        self.gs  = f.SinFunc("g", 3.0, 2.0, 'y')
        self.fc  = f.CosFunc("f", 1.0, 1.0)
        self.gc  = f.CosFunc("g", -3.0, -2.0, 'y')

    def test_01_call_sin(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.fs(0.0), 0.0)
        self.assertAlmostEqual(self.fs(math.pi/2), 1.0)
        self.assertAlmostEqual(self.fs(math.pi), 0.0)
        self.assertAlmostEqual(self.fs(1.0), 0.8414709848078965)
        self.assertAlmostEqual(self.gs(0.0), 0.0)
        self.assertAlmostEqual(self.gs(-math.pi/2), 0.0)
        self.assertAlmostEqual(self.gs(-math.pi), 0.0)
        self.assertAlmostEqual(self.gs(1.0), 2.727892280477045)

    def test_02_call_cos(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.fc(0.0), 1.0)
        self.assertAlmostEqual(self.fc(math.pi/2), 0.0)
        self.assertAlmostEqual(self.fc(math.pi), -1.0)
        self.assertAlmostEqual(self.gc(1.0), 1.2484405096414273)
        

    def test_03_str_sin(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.fs),'f(x)=sin(x)')
        self.assertEqual(str(self.gs),'g(y)=3.0sin(2.0y)')

    def test_04_str_cos(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.fc),'f(x)=cos(x)')
        self.assertEqual(str(self.gc),'g(y)=-3.0cos(-2.0y)')
    
    def test_05_derive_once(self):
        """
        Testet Ableitung.
        """
        fsd= self.fs.derive()
        gsd= self.gs.derive()
        fcd= self.fc.derive()
        gcd= self.gc.derive()
        self.assertTrue(type(fsd) is f.CosFunc)
        self.assertEqual(fsd.factor, 1.0)
        self.assertEqual(fsd.cos_factor, 1.0)
        self.assertEqual(fsd.operand, 'x')
        self.assertEqual(fsd._str_internal(), self.fc._str_internal())
        self.assertEqual(str(gsd), "g'(y)=6.0cos(2.0y)")
        self.assertEqual(str(fcd), "f'(x)=-sin(x)")
        self.assertEqual(str(gcd), "g'(y)=-6.0sin(-2.0y)")

    def test_06_derive_multiple(self):
        fsd= self.fs.derive().derive().derive().derive()
        fcd= self.fc.derive().derive().derive().derive()
        self.assertEqual(str(fsd), "f''''(x)=sin(x)")
        self.assertEqual(str(fcd), "f''''(x)=cos(x)")

        gsd= self.gs.derive().derive()
        gcd= self.gc.derive().derive()
        self.assertEqual(str(gsd), "g''(y)=-12.0sin(2.0y)")
        self.assertEqual(str(gcd), "g''(y)=12.0cos(-2.0y)")

class TestPowerFunc(unittest.TestCase):
    """
    Unittests für die Potenz-Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f  = f.PowerFunc("f", 1.0, 2.0)
        self.g  = f.PowerFunc("g", -3.0, 1.0)
        self.h  = f.PowerFunc("h", 0.5, 3.0, 'y')

    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(1.0), 1.0)
        self.assertAlmostEqual(self.f(3.0), 9.0)
        self.assertAlmostEqual(self.f(-2.5), 6.25)
        self.assertAlmostEqual(self.g(0.0), 0.0)
        self.assertAlmostEqual(self.g(2.0), -6.0)
        self.assertAlmostEqual(self.h(0.5), 0.0625)
        self.assertAlmostEqual(self.h(-1.5), -1.6875)
        
    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),'f(x)=x^2.0')
        self.assertEqual(str(self.g),'g(x)=-3.0x')
        self.assertEqual(str(self.h),'h(y)=0.5y^3.0')

    def test_03_derive_to_power(self):
        """
        Testet Ableitung auf eine andere Potenzfunktion.
        """
        fd= self.f.derive()
        hd= self.h.derive()
        hdd= hd.derive()

        self.assertTrue(type(fd) is f.PowerFunc)
        self.assertEqual(fd.factor, 2.0)
        self.assertEqual(fd.exponent, 1.0)
        self.assertEqual(fd.operand, 'x')
        self.assertEqual(str(fd), "f'(x)=2.0x")
        self.assertEqual(str(hd), "h'(y)=1.5y^2.0")
        self.assertEqual(str(hdd), "h''(y)=3.0y")

    def test_04_derive_to_const(self):
        """
        Testet Ableitung zur Konstantenfunktion.
        """
        gd= self.g.derive()
        gdd= self.g.derive().derive()
        fdd= self.f.derive().derive()
        self.assertTrue(type(gd) is f.ConstFunc)
        self.assertTrue(type(gdd) is f.ConstFunc)
        self.assertTrue(type(fdd) is f.ConstFunc)
        self.assertEqual(str(gd), "g'(x)=-3.0")
        self.assertEqual(str(gdd), "g''(x)=0.0")
        self.assertEqual(str(fdd), "f''(x)=2.0")

class TestSumFunc(unittest.TestCase):
    """
    Unittests für die zusammengestzte Summen-Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f  = f.SumFunc("f",[f.PowerFunc("", 1.0, 3.0), f.PowerFunc("", 1.0, 2.0), f.PowerFunc("", 1.0, 1.0)])
        self.g  = f.SumFunc("g",[f.PowerFunc("", 2.5, 2.0), f.PowerFunc("", -4.0, 1.0), f.ConstFunc("", 3.0)])
        self.h  = f.SumFunc("h", [f.SinFunc(), f.CosFunc("",3.0, 0.5), f.ConstFunc("", -1.0)])
        self.k  = f.SumFunc("k", [f.PowerFunc("", 1.0, 4.0), f.PowerFunc("", 5.0, 1.0)])

    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(2.0), 14.0)
        self.assertAlmostEqual(self.f(3.0), 39.0)
        self.assertAlmostEqual(self.g(0.0), 3.0)
        self.assertAlmostEqual(self.g(3.0), 13.5)
        self.assertAlmostEqual(self.g(-1.0), 9.5)
        self.assertAlmostEqual(self.h(math.pi), -1.0)
        self.assertAlmostEqual(self.h(math.pi/2), 2.121320343559643)
        
    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),"f(x)=(x^3.0+x^2.0+x)")
        self.assertEqual(str(self.g),"g(x)=(2.5x^2.0-4.0x+3.0)")
        self.assertEqual(str(self.h),"h(x)=(sin(x)+3.0cos(0.5x)-1.0)")

    def test_03_derive_once(self):
        """
        Testet einmalige Ableitung.
        """
        fd= self.f.derive()
        gd= self.g.derive()
        hd= self.h.derive()
        
        self.assertTrue(type(fd) is f.SumFunc)
        self.assertEqual(fd.factor, 1.0)
        self.assertEqual(fd.operand, 'x')
        self.assertIn("f'(x)=(", str(fd))
        self.assertIn("3.0x^2.0", str(fd))
        self.assertIn("2.0x", str(fd))
        self.assertIn("1.0", str(fd))
        self.assertAlmostEqual(fd(3.0), 34.0)
        
        self.assertEqual(len(gd.terms), 2)
        self.assertNotIn("0.0", str(gd))
        self.assertIn("4.0", str(gd))
        self.assertIn("5.0x", str(gd))
        self.assertAlmostEqual(gd(-2.0), -14.0)

        self.assertIn("cos(x)", str(hd))
        self.assertIn("-1.5sin(0.5x)", str(hd))
        self.assertAlmostEqual(hd(math.pi), -2.5)
        
        #self.assertEqual(str(fd), "f'(x)=2.0x")
        #self.assertEqual(str(hd), "h'(y)=1.5y^2.0")
        #self.assertEqual(str(hdd), "h''(y)=3.0y")

    def test_04_derive_multiple(self):
        """
        Testet mehrfache Ableitung und Wegfall von Konstanten.
        """
        
        fdd= self.f.derive().derive()
        fddd= self.f.derive().derive().derive()
        kdd= self.k.derive().derive()
        
        #print([str(ele) for ele in fdd.terms])

        self.assertEqual(len(fdd.terms), 2)
        self.assertIn("f''(x)=(", str(fdd))
        self.assertIn("6.0x", str(fdd))
        self.assertIn("+2.0", str(fdd))

        self.assertTrue(type(fddd) is f.ConstFunc)
        self.assertEqual(str(fddd), "f'''(x)=6.0")

        self.assertTrue(type(kdd) is f.PowerFunc)
        self.assertEqual(str(kdd), "k''(x)=12.0x^2.0")

class TestProdFunc(unittest.TestCase):
    """
    Unittests für die zusammengestzte Produkt-Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f = f.ProdFunc("f", f.PowerFunc("", 1.0, 2.0), f.SinFunc())
        self.g = f.ProdFunc("g", f.ExpFunc("", 2.0, 2.0), f.CosFunc("", 3.0))
        self.h = f.ProdFunc("h", f.SinFunc("", -1.0, 0.5), f.PowerFunc("", -3.0, 1.5))
        
    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(2.0), 3.637189707302727)
        self.assertAlmostEqual(self.f(3.0), 1.2700800725388048)
        self.assertAlmostEqual(self.g(2.0), -136.32508450571538)
        self.assertAlmostEqual(self.g(1.0), 23.953944290647627)
        self.assertAlmostEqual(self.g(-1.0), 0.43873179358835784)
        self.assertAlmostEqual(self.g(2.5), -713.4015292690831)

        
    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),"f(x)=x^2.0*sin(x)")
        self.assertEqual(str(self.g),"g(x)=6.0e^(2.0x)*cos(x)")
        self.assertEqual(str(self.h),"h(x)=3.0sin(0.5x)*x^1.5")

    def test_03_derive(self):
        """
        Testet einmalige Ableitung.
        """
        fd= self.f.derive()
        gd= self.g.derive()
        hd= self.h.derive()

        self.assertTrue(type(fd) is f.SumFunc)
        self.assertEqual(len(fd.terms), 2)
        self.assertIn("f'(x)=(", str(fd))
        self.assertIn("2.0x*sin(x)", str(fd))
        self.assertIn("x^2.0*cos(x)", str(fd))
        
        self.assertIn("12.0e^(2.0x)*cos(x)", str(gd))
        self.assertIn("-6.0e^(2.0x)*sin(x)", str(gd))
        self.assertAlmostEqual(gd(0.0), 12.0)

        self.assertIn("1.5cos(0.5x)*x^1.5", str(hd))
        self.assertIn("4.5sin(0.5x)*x^0.5", str(hd))
        self.assertAlmostEqual(hd(math.pi), 7.976042329074822)

class TestNestedFunc(unittest.TestCase):
    """
    Unittests für die verkettete Funktion.
    """
    def setUp(self):
        """
        Initialisiere Variablen für den Test.
        """
        self.f = f.NestedFunc("f", f.PowerFunc("", 1.0, 2.0), f.SinFunc())
        self.g = f.NestedFunc("g", f.CosFunc("", 3.0), f.ProdFunc("",  f.PowerFunc("", 1.0, 1.0), f.ExpFunc("", 1.0, 1.0)) )
        self.h = f.NestedFunc("h", f.ExpFunc("", 2.0, 1.0), f.SumFunc("", [f.SinFunc(), f.CosFunc()]) )
        
    def test_01_call(self):
        """
        Testet korrekte Berechnung.
        """
        self.assertAlmostEqual(self.f(0.0), 0.0)
        self.assertAlmostEqual(self.f(1.0), 0.7080734182735712)
        self.assertAlmostEqual(self.g(2.0), -1.7938600825330877)
        self.assertAlmostEqual(self.g(-1.0), 2.7992762267946256)
        self.assertAlmostEqual(self.h(-1.0), 1.479905895481268)
        self.assertAlmostEqual(self.h(0.0), 5.43656365691809)

        
    def test_02_str(self):
        """
        Testet korrekte Ausgabe als String.
        """
        self.assertEqual(str(self.f),"f(x)=sin(x)^2.0")
        self.assertEqual(str(self.g),"g(x)=3.0cos(x*e^x)")
        self.assertEqual(str(self.h),"h(x)=2.0e^((sin(x)+cos(x)))")

    def test_03_derive(self):
        """
        Testet einmalige Ableitung.
        """
        fd= self.f.derive()
        gd= self.g.derive()
        hd= self.h.derive()

        self.assertTrue(type(fd) is f.ProdFunc)
        self.assertEqual(str(fd), "f'(x)=2.0sin(x)*cos(x)")
        self.assertAlmostEqual(fd(1.0), 0.9092974268256818)
        
        self.assertEqual(str(gd), "g'(x)=-3.0sin(x*e^x)*(e^x+x*e^x)")
        self.assertAlmostEqual(gd(1.0), -6.699715904670079)

        self.assertEqual(str(hd), "h'(x)=2.0e^((sin(x)+cos(x)))*(cos(x)-sin(x))")
        self.assertAlmostEqual(hd(1.0), -2.398481179592898)


class TestIntegration(unittest.TestCase):
    """
    Erweiterte Unittests für numerische Integrationsmethoden.
    """

    def test_polynomial_integration(self):
        poly = f.PowerFunc("f", 2.0, 2.0)  # f(x) = 2x²
        self.assertAlmostEqual(poly.definite_integral(0, 2), 16/3, places=5)
        self.assertAlmostEqual(poly.untersumme(0, 2, 100), 16/3, delta=0.1)
        self.assertAlmostEqual(poly.obersumme(0, 2, 100), 16/3, delta=0.1)
        self.assertAlmostEqual(poly.trapezregel(0, 2, 100), 16/3, delta=0.01)

    def test_sinus_integration(self):
        sin_func = f.SinFunc("s", 2.0)  # f(x) = 2sin(x)
        expected = -2 * (math.cos(math.pi) - math.cos(0))  # = 4
        self.assertAlmostEqual(sin_func.definite_integral(0, math.pi), expected, places=5)
        self.assertAlmostEqual(sin_func.trapezregel(0, math.pi, 100), expected, delta=0.05)

    def test_constant_zero_function(self):
        zero_func = f.ConstFunc("z", 0.0)
        self.assertAlmostEqual(zero_func.definite_integral(-5, 5), 0.0)
        self.assertAlmostEqual(zero_func.trapezregel(-5, 5), 0.0)
        self.assertAlmostEqual(zero_func.obersumme(-5, 5), 0.0)
        self.assertAlmostEqual(zero_func.untersumme(-5, 5), 0.0)

    def test_exponential_integration(self):
        exp_func = f.ExpFunc("e", 1.0)
        expected = math.exp(1) - 1
        self.assertAlmostEqual(exp_func.definite_integral(0, 1), expected, delta=0.01)
        self.assertAlmostEqual(exp_func.trapezregel(0, 1, 100), expected, delta=0.02)

    def test_negative_interval(self):
        f1 = f.PowerFunc("f", 1.0, 3.0)  # x³
        val = f1.definite_integral(-1, 1)
        self.assertAlmostEqual(val, 0.0, delta=1e-10)  # ungerade Funktion über symmetrisches Intervall

    def test_piecewise_sum_function(self):
        sfunc = f.SumFunc("s", [f.ConstFunc("", 1.0), f.PowerFunc("", -1.0, 1.0)])  # f(x) = 1 - x
        expected = 0.5  # ∫(1 - x) dx von 0 bis 1 = 1 - 0.5 = 0.5
        self.assertAlmostEqual(sfunc.definite_integral(0, 1), expected, delta=0.01)

class TestNewtonMethod(unittest.TestCase):
    """
    Erweiterte Unittests für die Newton-Methode zur Nullstellenbestimmung.
    """

    def test_linear_function(self):
        linear = f.PowerFunc("f", 1.0, 1.0)  # f(x) = x
        zeros = linear.newton(-1, 1)
        self.assertTrue(any(abs(z) < 1e-5 for z in zeros))

    def test_quadratic_function_no_real_zeros(self):
        quad = f.SumFunc("f", [f.PowerFunc("", 1.0, 2.0), f.ConstFunc("", 1.0)])  # x² + 1
        zeros = quad.newton(-5, 5)
        self.assertEqual(len(zeros), 0)

    def test_trig_function_multiple_zeros(self):
        sin_func = f.SinFunc("s", 1.0)
        zeros = sin_func.newton(0, 4 * math.pi)
        self.assertTrue(len(zeros) >= 3)
        self.assertTrue(any(abs(z) - math.pi < 0.1 for z in zeros))

    def test_sin_minus_one_zero(self):
        shifted_sin = f.SumFunc("f", [f.SinFunc(), f.ConstFunc("", -1.0)])  # sin(x) - 1
        zeros = shifted_sin.newton(0, 2 * math.pi)
        self.assertTrue(any(abs(z - math.pi/2) < 0.2 for z in zeros))  # pi/2 ≈ Nullstelle

    def test_nested_function(self):
        inner = f.PowerFunc("", 1.0, 1.0)
        outer = f.SinFunc()
        nested = f.NestedFunc("n", outer, inner)  # sin(x)
        zeros = nested.newton(0, 2 * math.pi)
        self.assertTrue(len(zeros) >= 2)
        self.assertTrue(any(abs(z - math.pi) < 0.1 for z in zeros))

    def test_zero_derivative_handling(self):
        flat = f.ConstFunc("c", 5.0)
        zeros = flat.newton(-10, 10)
        self.assertEqual(len(zeros), 0)  # keine Nullstelle

    def test_multiple_close_zeros(self):
        func = f.ProdFunc("p", f.PowerFunc("", 1.0, 1.0), f.PowerFunc("", 1.0, 2.0))  # x * x² = x³
        zeros = func.newton(-0.5, 0.5)
        self.assertTrue(len(zeros) >= 1)
        self.assertTrue(any(abs(z) < 1e-3 for z in zeros))

def runIntegrationTests():
    # Integration
    suite = unittest.TestSuite()

    suite.addTest(TestIntegration("test_polynomial_integration"))
    suite.addTest(TestIntegration("test_sinus_integration"))
    suite.addTest(TestIntegration("test_constant_zero_function"))
    suite.addTest(TestIntegration("test_exponential_integration"))
    suite.addTest(TestIntegration("test_negative_interval"))
    suite.addTest(TestIntegration("test_piecewise_sum_function"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

def runNewtonTests():
    # Newton
    suite = unittest.TestSuite()

    suite.addTest(TestNewtonMethod("test_linear_function"))
    suite.addTest(TestNewtonMethod("test_quadratic_function_no_real_zeros"))
    suite.addTest(TestNewtonMethod("test_trig_function_multiple_zeros"))
    suite.addTest(TestNewtonMethod("test_sin_minus_one_zero"))
    suite.addTest(TestNewtonMethod("test_nested_function"))
    suite.addTest(TestNewtonMethod("test_zero_derivative_handling"))
    suite.addTest(TestNewtonMethod("test_multiple_close_zeros"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

def runAllTests():
    #Durchführung der TestsConstFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestConstFunc("test_01_call"))
    suite.addTest(TestConstFunc("test_02_str"))
    suite.addTest(TestConstFunc("test_03_derive"))

    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsExpFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestExpFunc("test_01_call"))
    suite.addTest(TestExpFunc("test_02_str"))
    suite.addTest(TestExpFunc("test_03_derive"))

    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsSinCos
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestSinCos("test_01_call_sin"))
    suite.addTest(TestSinCos("test_02_call_cos"))
    suite.addTest(TestSinCos("test_03_str_sin"))
    suite.addTest(TestSinCos("test_04_str_cos"))
    suite.addTest(TestSinCos("test_05_derive_once"))
    suite.addTest(TestSinCos("test_06_derive_multiple"))

    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsPowerFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestPowerFunc("test_01_call"))
    suite.addTest(TestPowerFunc("test_02_str"))
    suite.addTest(TestPowerFunc("test_03_derive_to_power"))
    suite.addTest(TestPowerFunc("test_04_derive_to_const"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsSumFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestSumFunc("test_01_call"))
    suite.addTest(TestSumFunc("test_02_str"))
    suite.addTest(TestSumFunc("test_03_derive_once"))
    suite.addTest(TestSumFunc("test_04_derive_multiple"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsProdFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestProdFunc("test_01_call"))
    suite.addTest(TestProdFunc("test_02_str"))
    suite.addTest(TestProdFunc("test_03_derive"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

    #Durchführung der TestsNestedFunc
    suite = unittest.TestSuite()

    #Hier können einzelne Tests auskommentiert werden
    suite.addTest(TestNestedFunc("test_01_call"))
    suite.addTest(TestNestedFunc("test_02_str"))
    suite.addTest(TestNestedFunc("test_03_derive"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # ----------------- Neue Tests ---------------
    # Integration
    suite = unittest.TestSuite()

    suite.addTest(TestIntegration("test_polynomial_integration"))
    suite.addTest(TestIntegration("test_sinus_integration"))
    suite.addTest(TestIntegration("test_constant_zero_function"))
    suite.addTest(TestIntegration("test_exponential_integration"))
    suite.addTest(TestIntegration("test_negative_interval"))
    suite.addTest(TestIntegration("test_piecewise_sum_function"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # Newton
    suite = unittest.TestSuite()

    suite.addTest(TestNewtonMethod("test_linear_function"))
    suite.addTest(TestNewtonMethod("test_quadratic_function_no_real_zeros"))
    suite.addTest(TestNewtonMethod("test_trig_function_multiple_zeros"))
    suite.addTest(TestNewtonMethod("test_sin_minus_one_zero"))
    suite.addTest(TestNewtonMethod("test_nested_function"))
    suite.addTest(TestNewtonMethod("test_zero_derivative_handling"))
    suite.addTest(TestNewtonMethod("test_multiple_close_zeros"))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)