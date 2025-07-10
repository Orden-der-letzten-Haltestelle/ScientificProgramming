import matplotlib.pyplot as plt
import math

class Plottable():
    '''
    Abstrakte Basisklasse, um Funktionen darstellen zu können.
    '''
    def sample(self, minimum : float, maximum : float, samples : int) -> [float]:
        sample_list = [self(minimum + (i / (samples -1.0) * (maximum - minimum))) for i in range(0, samples)]
        return sample_list


    @staticmethod
    def _get_x_values(minimum : float, maximum : float, samples : int) -> [float]:
        return [minimum + (i / (samples - 1.0) * (maximum - minimum)) for i in range(0, samples)]
    
    @staticmethod
    def multi_plot(plottables : [], minimum : float, maximum : float, samples : int = 100, area : bool = False, n : int = 7, mode : int = 0) -> None:
        plt.figure(figsize=(8, 6))
        x = Plottable._get_x_values(minimum, maximum, samples)
        for plottable in plottables:
            y = plottable.sample(minimum, maximum, samples)
            plt.plot(x, y, label=f"{plottable}")

            if area:
                width = (maximum - minimum) / n
                xs = [minimum + i * width for i in range(n + 1)]
                ys = [plottable(xi) for xi in xs]

                if mode == 2: # Trapez
                    for i in range(n):
                        verts = [(xs[i], 0), (xs[i], ys[i]), (xs[i+1], ys[i+1]), (xs[i+1], 0)]
                        plt.gca().add_patch(plt.Polygon(verts, closed=True, alpha=0.3, color='orange'))

                else:
                    for i in range(n):
                        xi = xs[i]
                        if mode == 0:  # Untersumme
                            yi = min(ys[i], ys[i+1])
                        elif mode == 1:  # Obersumme
                            yi = max(ys[i], ys[i+1])

                        plt.gca().add_patch(plt.Rectangle((xi, 0), width, yi,
                                                        alpha=0.3, edgecolor='blue', facecolor='cyan'))
                        
        Plottable._configure_plot_and_show()

    def plot(self, minimum : float, maximum : float, samples : int = 100) -> None: 
        x = Plottable._get_x_values(minimum, maximum, samples)
        y = self.sample(minimum, maximum, samples)
        plt.plot(x, y, label=f"{self}")
        Plottable._configure_plot_and_show()

    @staticmethod
    def _configure_plot_and_show() -> None:
        # Axes and labels
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def plot_with_random_points(self, minimum: float, maximum: float, yBorderMin, yBorderMax, points : []) -> None:
        x_vals = Plottable._get_x_values(minimum, maximum, 100)
        y_vals = self.sample(minimum, maximum, 100)

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="Funktion")
        
        plt.axhline(y=yBorderMin, color='purple', linestyle='--')
        plt.axhline(y=yBorderMax, color='purple', linestyle='--')

        rand_x = points[:, 0]
        rand_y = points[:, 1]

        plt.scatter(rand_x, rand_y, color='red', s=10, label="Zufälliger Punkt")

        Plottable._configure_plot_and_show()

#====================================================================
# Abstrakte Basisklasse der Funktionen

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import copy

class MFunc(ABC, Plottable):
    '''
    Abstrakte Basisklasse für Funktionen.
    '''
    def __init__(self, name: str = '', factor: float = 1.0, operand : str ="x"):
        '''
        Konstruktor mit Funktionsname (z.B. f), konstantem Faktor, und Operand.
        '''
        self.name = name
        self.factor = factor
        self.operand = operand

    def _factor_str(self, factor: float) -> str:
        '''
        Hilfsmethode, die einen Faktor in einen String umwandelt:
        Faktor 1.0 kann weggelassen werden.
        Faktor -1.0 kann als Minus geschrieben werden.
        Alle anderen Faktoren werden als Zahl angegeben.
        '''
        return f"{'' if factor == 1.0 else '-' if factor == -1.0 else factor}"
    
    @abstractmethod
    def _str_internal(self) -> str:
        '''
        Wandelt die eigentliche Funktion in einen String um, ohne Faktor und Funktionssymbol.
        z.B. __str__: f(x)=3.0cos(x) -> _str_internal: cos(x)
        '''
        pass

    def __str__(self) -> str:
        '''
        Wandelt die Funktion in einen menschenlesbaren String um.
        '''
        return f"{self.name}({self.operand})={self._str_internal()}"

    @abstractmethod
    def _call_internal(self, x: float) -> float:
        '''
        Berechnet die eigentliche Funktion ohne Faktor.
        Ein Error wird als NaN (not a number) interpretiert.
        '''
        pass
    
    def __call__(self, x: float) -> float:
        '''
        Berechnet den Funktionswert an Stelle x.
        '''
        result = 0
        try:
            result = self._call_internal(x)
        except: #z.B. durch Teilen durch Null
            #print(f"Error at {self} {x}")
            return math.nan
        return self.factor * result

    @abstractmethod
    def derive(self):
        '''
        Gibt die Ableitungsfunktion dieser FUnktion zurück.
        '''
        pass

    # @abstractmethod
    def integrate(self):
        '''
        Gibt eine Stammfunktion dieser Funktion zurück. c = 0
        '''
        pass

    def definite_integral(self, a, b):
        sf = self.integrate()
        return  sf(b) - sf(a)

    def newton(self, start, end):
        '''
        Gibt die Nullstellen der Funktion numerisch wieder
        '''
        table = []
        nullstellen = []
        current = start

        step = abs(end - start) / 100
        while (current <= end):
            if self(current) == 0:
                nullstellen.append(current)
            
            elif self(current) * self(current + step) < 0:
                table.append(current)
            current += step

        if len(table) != 0:
            for ele in table:
                xn = ele - 1
                x0 = ele
                for l in range(31):
                    if self.derive()(x0) == 0: 
                        xn = x0 - self(x0)/(self.derive()(x0)+1e-5)
                    else:
                        xn = x0 - self(x0)/self.derive()(x0)
                    x0 = xn

                nullstellen.append(x0)
        
        nullstellen.sort()
        #print("Nullstellen:", nullstellen)
        return nullstellen

    def definitions_luecken(self, start, end):
        luecken = [] 
        step_factor = 0.1
        for i in range(start * int(1/step_factor), end * int(1/step_factor)):
            if math.isnan(self(i)):
                luecken.append(i)

        return luecken

    def __untersumme(self, start, end, n):
        sum = 0
        width = abs(start - end) / n
        for i in range(n):
            x = start + i * width
            if not math.isnan(self(x)):
                sum += self(x) * width
        
        return abs(sum)

    def untersumme(self, start, end, n = 10):
        '''
        Für die numerische Integration mit der Untersumme
        '''
        sum = 0
        nullstellen = self.newton(start, end)

        for i, ele in enumerate(nullstellen):
            if ele != start and i == 0:
                sum += self.__untersumme(start, ele, n)
            
            if i < len(nullstellen)-1:
                sum += self.__untersumme(ele, nullstellen[i+1], n)
            else: 
                sum += self.__untersumme(ele, end, n)

        return sum

    def __obersumme(self, start, end, n):
        sum = 0
        width = abs(start - end) / n
        for i in range(1, n+1):
            x = start + i * width
            if not math.isnan(self(x)):
                sum += self(x) * width
        
        return abs(sum)

    def obersumme(self, start, end, n = 10):
        '''
        Für die numerische Integration mit der Obersumme
        '''
        sum = 0
        nullstellen = self.newton(start, end)

        for i, ele in enumerate(nullstellen):
            if ele != start and i == 0:
                print("Obersumme 1", sum, start, ele)
                sum += self.__obersumme(start, ele, n)

            if i < len(nullstellen)-1:
                print("Obersumme 2", sum, ele, nullstellen[i+1]), 
                sum += self.__obersumme(ele, nullstellen[i+1], n)
            else: 
                print("Obersumme 3", sum, ele, end)
                sum += self.__obersumme(ele, end, n)

        return sum

    def __trapezregel(self, start, end, n):
        sum = 0
        width = abs(start - end) / n
        for i in range(n):
            x1 = start + i * width
            x2 = start + (i+1) * width
            if not (math.isnan(self(x1)) or math.isnan(self(x2))):
                sum += 0.5 * (self(x1) + self(x2)) * width
            
        return abs(sum)

    def trapezregel(self, start, end, n = 10):
        '''
        Für die numerische Integration mit der Trapezregel
        '''
        sum = 0
        nullstellen = self.newton(start, end)

        for i, ele in enumerate(nullstellen):
            if ele != start and i == 0:
                sum += self.__trapezregel(start, ele, n)
            
            if i < len(nullstellen)-1:
                sum += self.__trapezregel(ele, nullstellen[i+1], n)
            else: 
                sum += self.__trapezregel(ele, end, n)

        return sum

    def clone(self):
        '''
        Erzeugt eine Kopie der funktion und aller ihrer Attribute.
        '''
        return copy.deepcopy(self)
    
    def call_verbose(self, x: float):
        '''
        Berechnet den Funktionswert an Stelle x und gibt ihn schön als String formatiert zurück.
        '''
        return f"{self.name}({x})={self(x)}"

#====================================================================
# Verschiedene Funktionen

class ConstFunc(MFunc):
    '''
    Konstante welche immer den Faktor zurückgibt und die Ableitung 0.0 hat
    '''

    def _str_internal(self) -> str:
        return f"{self.factor}"

    def _call_internal(self, x: float) -> float:
        '''
        Gibt immer 1 zurück, der Faktor kommt später hinzu.
        '''
        return 1

    def derive(self):
        return ConstFunc(self.name + "'", 0.0, self.operand)

    def integrate(self):
        return PowerFunc(self.name.upper(), self.factor, 1)

class ExpFunc(MFunc):
    '''
    Exponentialfunktion factor * e^{exp_factor * x}
    '''
    
    def __init__(self, name: str = '', factor: float = 1.0, exp_factor: float = 1.0, operand : str ="x"):
        super().__init__(name, factor, operand)
        self.exp_factor = exp_factor
        
    def _str_internal(self) -> str:
        exponent = f"{self._factor_str(self.exp_factor)}{self.operand}"
        if(abs(self.exp_factor) != 1.0 or len(self.operand) > 1):
            if not isinstance(exponent, SumFunc):
                exponent = f"({exponent})"
        return f"{self._factor_str(self.factor)}e^{exponent}"

    def _call_internal(self, x: float) -> float:
        return math.pow(math.e, self.exp_factor*x)

    def derive(self) -> MFunc:
        return ExpFunc(self.name+"'", self.factor * self.exp_factor, self.exp_factor, self.operand)

    def  integrate(self):
        return ExpFunc(self.name.upper(), self.factor * (1/self.exp_factor), self.exp_factor)

class SinFunc(MFunc):
    '''
    Sinusfunktion in der From: factor * sin(sin_factor * x)
    '''

    def __init__(self, name: str = '', factor: float = 1.0, sin_factor: float = 1.0, operand : str ="x"):
        super().__init__(name, factor, operand)
        self.sin_factor = sin_factor
        
    def _str_internal(self) -> str:
        return f"{self._factor_str(self.factor)}sin({self._factor_str(self.sin_factor)}{self.operand})"

    def _call_internal(self, x: float) -> float:
        return math.sin(self.sin_factor * x)

    def derive(self) -> MFunc:
        return CosFunc(self.name + "'", self.factor * self.sin_factor, self.sin_factor, self.operand) 

    def integrate(self):
        return CosFunc(self.name.upper(), self.factor * (-1/self.sin_factor), self.sin_factor)
    

class CosFunc(MFunc):
    '''
    Kosinusfunktion in der From: factor * cos(cos_factor * x)
    '''

    def __init__(self, name: str = '', factor: float = 1.0, cos_factor: float = 1.0, operand : str ="x"):
        super().__init__(name, factor, operand)
        self.cos_factor = cos_factor
        
    def _str_internal(self) -> str:
        return f"{self._factor_str(self.factor)}cos({self._factor_str(self.cos_factor)}{self.operand})"

    def _call_internal(self, x: float) -> float:
        return math.cos(self.cos_factor * x)

    def derive(self) -> MFunc:
        return SinFunc(self.name + "'", (-1) * self.factor * self.cos_factor, self.cos_factor, self.operand)

    def integrate(self):
        return SinFunc(self.name.upper(), self.factor * (1/self.cos_factor), self.cos_factor)

class PowerFunc(MFunc):
    '''
    Funktionen der Form: factor * x^{exponent}
    '''
    
    def __init__(self, name: str = '', factor: float = 1.0, exponent: float = 1.0, operand : str ="x"):
        super().__init__(name, factor, operand)
        self.exponent = exponent
        
    def _str_internal(self) -> str:
        return f"{self._factor_str(self.factor)}{self.operand}{'' if self.exponent == 1 else f'^{self.exponent}'}"

    def _call_internal(self, x: float) -> float:
        #print("[DEBUG Power]", self._str_internal(self), x, self.exponent)
        return math.pow(x, self.exponent)

    def derive(self) -> MFunc:
        if self.exponent == 1: 
            return ConstFunc(self.name + "'", self.factor, self.operand)
        return PowerFunc(self.name + "'", self.factor * self.exponent, self.exponent - 1, self.operand) 

    def integrate(self):
        if self.exponent == -1:
            print("x^-1 Klappt leider nicht!")
            # Wäre ln(abs(x)), es gibt aber keine Klasse dafür
            return None
        return PowerFunc(self.name.upper(), self.factor * (1/(self.exponent+1)), self.exponent+1)

#====================================================================
# Verschiedene Kombination von Funktionen

class SumFunc(MFunc):
    '''
    Summen von Funktionen der Form: MFunc + MFunc + ...
    '''

    def __init__(self, name: str, terms: [MFunc]):
        super().__init__(name)
        self.terms = []
        for term in terms:
            self.terms.append(term.clone())
        
    def _str_internal(self) -> str:
        out = "("

        for i, ele in enumerate(self.terms):
            if ele.factor != 0:
                if ele.factor > 0 and i != 0: 
                    out += "+"
                out += ele._str_internal()

        return out + ")"

    def _call_internal(self, x: float) -> float:
        sum = 0
        for ele in self.terms:
            sum += ele(x)

        return sum

    def derive(self) -> MFunc:
        newTerms = []
        for ele in self.terms:
            if not ("0.0" in str(ele.derive())): 
                newTerms.append(ele.derive())
        
        if len(newTerms) == 1: 
            if type(newTerms[0]) == ConstFunc: 
                return ConstFunc(self.name + "'", newTerms[0].factor, newTerms[0].operand) 
            elif type(newTerms[0]) == PowerFunc: 
                return PowerFunc(self.name + "'", newTerms[0].factor, newTerms[0].exponent, newTerms[0].operand)
        return SumFunc(self.name + "'", newTerms)
        
    def __len__(self):
        return len(self.terms)

    def integrate(self):
        newTerms = []
        for ele in self.terms:
            newTerms.append(ele.integrate())
        return SumFunc(self.name.upper(), newTerms)

class ProdFunc(MFunc):
    '''
    Produkt von zwei Funktionen: MFunc * MFunc
    Wofür ist der factor?
    '''

    def __init__(self, name: str, left: MFunc, right: MFunc, factor: float = 1.0):
        super().__init__(name, factor)
        self.left = left
        self.right = right

    def _str_internal(self) -> str:
        if self.right.factor != 1: 
            self.left.factor *= self.right.factor
            self.right.factor = 1
        if self.left._str_internal() == "1.0":
            return f"{self.right._str_internal()}"
        return f"{self.left._str_internal()}*{self.right._str_internal()}"

    def _call_internal(self, x: float) -> float:
        return self.left(x) * self.right(x) 

    def derive(self) -> MFunc:
        return SumFunc(self.name + "'", [ProdFunc("", self.left.derive(), self.right, self.factor), ProdFunc("", self.left, self.right.derive(), self.factor)])

    def integrate(self):
        #Nicht nötig
        return super().integrate()

class NestedFunc(MFunc):
    '''
    Verkettete Funktionen der Form: MFunc(MFunc)
    Wofür ist der factor?
    '''

    def __init__(self, name: str, outer: MFunc, inner: MFunc, factor: float = 1.0):
        super().__init__(name, factor)
        self.outer = outer
        self.inner = inner

    def _str_internal(self) -> str:
        self.outer.operand = self.inner._str_internal()
        return self.outer._str_internal()

    def _call_internal(self, x: float) -> float:
        return self.outer(self.inner(x))

    def derive(self) -> MFunc:  
        d_outer = self.outer.derive()
        d_inner = self.inner.derive()
        new_outer = NestedFunc("", d_outer, self.inner.clone())  # Außen abgeleitet, innen gleich
        return ProdFunc(self.name + "'", new_outer, d_inner, self.factor)
    
    def integrate(self):
        #Nicht nötig
        return super().integrate()