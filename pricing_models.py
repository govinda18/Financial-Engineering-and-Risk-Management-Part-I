import pandas as pd
import numpy as np
import risk_kit as rk
from abc import ABC


class BinomialTree(ABC):
    """
	Implements an abstract class for the mulit-period binomial pricing models.
	Uses the Lattice objects for the nodes of the tree structure.

	Parameters:
	----------

	n: int
		The number of periods for which the tree is to be made. 
		Also equivalent to the depth of the tree.

    q: float
        The probability of the security going upwards.
	"""

    @property
    def n(self):
        """
		Gets the the number of periods in the model.
		"""
        return self._n

    @n.setter
    def n(self, val):
        self._n = val
    

    @property
    def q(self):
        """
        Gets the probability of a security going up.
        """
        return self._q
    
    @q.setter
    def q(self, val):
        self._q = val

    @property
    def tree(self):
        """
		Gets the binomial tree with node as Lattice objects

        The tree for a n period model is returned in the form of a matrix as - 
        [[S0],
        [[dS0,  uS0],
        [[d2S0, duS0,   u2S0],
        .
        .
        [[dnS0, d(n-1)u1S0, ..., unS0]]                
		"""
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree


    def printtree(self):
        """
		Prints the prices of the binomial pricing tree.
		"""
        for i in range(self.n + 1):
            print(self.tree[i][:i+1])

    def __init__(self, n, q=0.5):
        """
		Initializes the data descriptors from the given parameters.
		"""

        self.n = n

        self.q = q

        self.tree = np.zeros([self.n + 1, self.n + 1])


class StockPricing(BinomialTree):
    """
    Implements the binomial stock pricing model. 
    Inherits the BinomialTree class.

    Parameters:
    ----------

    S0: float
        The initial price of the security

    u: float
        The upward drift of the security

    d: float
        The downward drift of the security

    c: float
        The dividend paid by the security
    """

    __doc__ += BinomialTree.__doc__

    @property
    def S0(self):
        return self._S0
    
    @S0.setter
    def S0(self, val):
        self._S0 = val

    @property
    def u(self):
        return self._u
    
    @u.setter
    def u(self, val):
        self._u = val

    @property
    def d(self):
        return self._d
    
    @d.setter
    def d(self, val):
        self._d = val

    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, val):
        self._c = val

    def _constructTree(self):
        """
        Constructs the pricing of the binomial model for n periods.
        """
        for i in range(self.n + 1):
            for j in range(i + 1):
                price = self.S0 * (self.u ** j) * (self.d ** (i - j))
                self.tree[i, j] = price

    def __init__(self, n, S0, u, d, c=0.0):
        """
        Initializes the binomail model for the corresponding parameters.
        """

        super().__init__(n)

        self.S0 = S0

        self.u = u

        self.d = d

        self.c = c

        self._constructTree()


class FuturesPricing(BinomialTree):
    """
    Implements a futures pricing model based on the binomial model.
    Inherits the BinomialTree class.

    Parameters:
    ----------
    
    n: int
        The period of the futures contract

    model: BinomialTree
        The underlying security pricing from which the futures contract is derived.

    q: float
        The probability of an upward move.

    unpaid_coupon: float
        The amount which the underlying security earns at the end of the contract but is not 
        paid to the long position holder in the contract.
        The contract is executed immeditately after the dividend/coupon is paid.
    """

    __doc__ += BinomialTree.__doc__


    @property
    def price(self):
        return self.tree[0, 0]    

    def _constructTree(self, model, coupon):
        """
        Recomputes the prices from the given model's spot prices for futures pricing.
        """

        for i in range(self.n, -1, -1):
            if i == self.n:
                self.tree[i] = model.tree[i, :(i + 1)] - coupon
            else:
                for j in range(i + 1):
                    childd = self.tree[i + 1, j]
                    childu = self.tree[i + 1, j + 1]

                    self.tree[i, j] = self.q * childu + (1 - self.q) * childd


    def __init__(self, n, model, q, unpaid_coupon=0.0):

        super().__init__(n, q)

        self._constructTree(model, unpaid_coupon)


class OptionsPricing(BinomialTree):
    """
	Implements a binomial tree based option pricing model.
    Inherits the BinomialTree class.

    Parameters
    ----------
    model: BinomialTree
        The underlying security model from which the options contract is derived.

    K: float
        The strike price of the option contract.

    r: float / BinomialTree
        The rate of interest to be used. Should be a scalar if fixed and a binomial model otherwise.

    is_call: bool
        Sets to True if the option is call and False if the option is put. Defaults to True,

    is_american: bool
        Sets to True if the option is American and False if the option is European. Defaults to False.
	"""

    __doc__ += BinomialTree.__doc__

    @property
    def K(self):
        """
        Represents the strike price of the options contract.
        """
        return self._K

    @K.setter
    def K(self, val):
        self._K = val

    @property
    def multiplier(self):
        """
        The multiplier to be used for call and put option pricing.
        Sets to 1 for call options and -1 for put options.
        """
        return self._multiplier

    @multiplier.setter
    def multiplier(self, val):
        self._multiplier = val

    @property
    def is_american(self):
        """
        Represents if the option security is american or european.
        """
        return self._is_american

    @is_american.setter
    def is_american(self, val):
        self._is_american = val

    @property
    def price(self):
        """
        Returns the current price of the option.
        """
        return self.tree[0, 0]

    @property
    def early_exercise(self):
        """
        Gets the details of early exercise of options.

        Returns a list of dictionaries sorted by time consisting of all the possible times
        when early exercise of options can be more beneficial.
        """
        result = []
        for time, no, early_ex, hold in sorted(self._early_exercise):
            data = {
                'Time': time,
                'Current Premium': early_ex,
                'Hold': hold,
            }
            result.append(data)

        return result
    

    def _constructTree(self, model, r):
        """
        Computes the option prices from the given pricing model and rate of interest.
        """

        if isinstance(r, int) or isinstance(r, float):
            rate = np.empty([self.n + 1, self.n + 1])
            rate.fill(r)
        else:
            rate = r.tree

        for i in range(self.n, -1, -1):
            if i == self.n:
                for j in range(i + 1):
                    self.tree[i, j] = max(
                        0, self.multiplier * (model.tree[i, j] - self.K)
                    )
            else:
                for j in range(i + 1):
                    childu = self.tree[i + 1, j + 1]
                    childd = self.tree[i + 1, j]

                    # Expected call option permium if portfolio is held
                    hold = (self.q * childu + (1 - self.q) * childd) / (1 + rate[i, j])

                    # Call option premium if portfolio is exercised
                    # Can be done only in the case of american options
                    early_ex = max(0, self.multiplier * (model.tree[i, j] - self.K))

                    if early_ex > hold:
                        self._early_exercise.append((i, j, early_ex, hold))

                    self.tree[i, j] = max(hold, early_ex) if self.is_american else hold


    def __init__(self, n, model, r, q, K, is_call=True, is_american=False):
        """
        Initializes the black scholes model and other parameters from the given parameters.
        """
        super().__init__(n, q)

        self.K = K

        self.multiplier = 1 if is_call else -1

        self.is_american = is_american

        self._early_exercise = []

        self._constructTree(model, r)


class BondPricing(BinomialTree):
    """
    Implements the binomial bond pricing model.
    Inherits the BinomialTree class.

    Parameters:
    ----------

    F: float
        The face value of the bond.

    u: float
        The factor by which the bond price goes up.

    d: float
        The factor by which the bond price goes down.

    c: float
        The coupon rate of the bond. Defaults to zero assuming zero coupon bond.

    """

    __doc__ += BinomialTree.__doc__

    @property
    def F(self):
        return self._F
    
    @F.setter
    def F(self, val):
        self._F = val

    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, val):
        self._c = val

    @property
    def price(self):
        return self.tree[0, 0]

    def _constructTree(self, rate):
        """
        Constructs the tree for bond pricing for n periods.
        """
        coupon = self.F * self.c

        self.tree[self.n] = np.repeat(self.F + coupon, self.n + 1)
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                childd = self.tree[i + 1, j]
                childu = self.tree[i + 1, j + 1]

                price = coupon + (self.q * childu + (1 - self.q) * childd) / (1 + rate[i, j])
                self.tree[i, j] = price


    def __init__(self, n, F, q, r, c=0.0):
        """
        Initializes the bond pricing model from the given parameters.
        """
        super().__init__(n, q)

        self.F = F

        self.c = c

        self.r = r

        self._constructTree(r.tree)


class ForwardsPricing(BinomialTree):
    """
    Implements a forwards pricing model based on the binomial model.
    Inherits the BinomialTree class.

    Parameters:
    ----------
    
    n: int
        The period of the futures contract

    model: BinomialTree
        The underlying security pricing from which the futures contract is derived.

    q: float
        The probability of an upward move.

    r: float / BinomialTree
        The rate of interest to be used. Should be a scalar if fixed and a binomial model otherwise.

    unpaid_coupon: float
        The amount which the underlying security earns at the end of the contract but is not 
        paid to the long position holder in the contract.
        The contract is executed immeditately after the dividend/coupon is paid.
    """

    __doc__ += BinomialTree.__doc__

    @property
    def r(self):
        """
        The rate of interest.
        """
        return self._r

    @r.setter
    def r(self, val):
        self._r = val

    @property
    def price(self):
        """
        Gets the price of the forward contract on the underlying security.
        """
        zcb_n = BondPricing(self.n, 1, self.q, self.r).price
        return self.tree[0, 0]  / zcb_n   

    def _constructTree(self, model, r, coupon):
        """
        Recomputes the prices from the given model's spot prices for futures pricing.
        """

        if isinstance(r, int) or isinstance(r, float):
            rate = np.empty([self.n + 1, self.n + 1])
            rate.fill(r)
        else:
            rate = r.tree

        for i in range(self.n, -1, -1):
            if i == self.n:
                self.tree[i] = model.tree[i, :(i + 1)] - coupon
            else:
                for j in range(i + 1):
                    childd = self.tree[i + 1, j]
                    childu = self.tree[i + 1, j + 1]

                    self.tree[i, j] = (self.q * childu + (1 - self.q) * childd) / (1 + rate[i, j])


    def __init__(self, n, model, q, r, unpaid_coupon=0.0):

        super().__init__(n, q)

        self.r = r

        self._constructTree(model, r, unpaid_coupon)


class SwapsPricing(BinomialTree):
    """
    Implements a swap pricing model based on the binomial model.
    Inherits the BinomialTree class.
    
    The model assumes the last exchange is executed at n + 1 period.

    Parameters:
    ----------
    
    fixed_rate: float
        The fixed rate of interest to be paid/recieved in the swap contract

    start_time: int
        The period from which the exchange starts

    is_long: bool
        The type of position to be modeled, long or short.
        Long position refers to paying the fixed  interest rate 
        while short refers to paying the floating rates.

    r: BinomialTree
        The rate model for varying interest rates
    """

    __doc__ += BinomialTree.__doc__

    @property
    def fixed_rate(self):
        return self._fixed_rate

    @fixed_rate.setter
    def fixed_rate(self, val):
        self._fixed_rate = val

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, val):
        self._start_time = val

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, val):
        self._multiplier = val

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val):
        self._r = val

    @property
    def price(self):
        return self.tree[0, 0]

    def _constructTree(self, r):
        """
        Constructs the binomial tree for pricing the swaps.
        """
        rate = r.tree

        for i in range(self.n, -1, -1):
            if i == self.n:
                self.tree[i] = (rate[i, :(i + 1)] - self.fixed_rate) * self.multiplier / (rate[i, :(i + 1)] + 1)
            else:
                for j in range(i + 1):
                    childd = self.tree[i + 1, j]
                    childu = self.tree[i + 1, j + 1]

                    value = (self.q * childu + (1 - self.q) * childd) / (1 + rate[i, j])

                    if i >= self.start_time - 1:
                        payment = ((rate[i, j] - self.fixed_rate) * self.multiplier) / (1 + rate[i, j])
                        value += payment

                    self.tree[i, j] = value


    def __init__(self, n, q, fixed_rate, start_time, is_long, r):
        """
        Initializes the model based on the given parameters.
        """

        super().__init__(n, q)

        self.fixed_rate = fixed_rate

        self.start_time = start_time

        self.multiplier = 1 if is_long else -1

        self.r = r

        self._constructTree(r)
