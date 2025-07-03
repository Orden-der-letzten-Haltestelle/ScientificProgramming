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
    def multi_plot(plottables : [], minimum : float, maximum : float, samples : int = 100) -> None:
        plt.figure(figsize=(8, 6))
        x = Plottable._get_x_values(minimum, maximum, samples)
        for plottable in plottables:
            y = plottable.sample(minimum, maximum, samples)
            plt.plot(x, y, label=f"{plottable}")
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
            print(f"Error at {self} {x}")
            return math.nan
        return self.factor * result

    @abstractmethod
    def derive(self):
        '''
        Gibt die Ableitungsfunktion dieser FUnktion zurück.
        '''
        pass

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