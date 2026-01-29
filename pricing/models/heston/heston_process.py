from processes.base_processes import StochasticProcess
import numpy as np

class HestonProcess(StochasticProcess):
    __slots__ = ('s0_', 'v0_', 'r_', 'q_', 'kappa_', 'theta_', 'sigma_', 'rho_')

    def __init__(self, s0, v0, r, q, kappa, theta, sigma, rho):
        self.s0_ = s0
        self.v0_ = v0
        self.r_ = r
        self.q_ = q
        self.kappa_ = kappa
        self.theta_ = theta
        self.sigma_ = sigma
        self.rho_ = rho

    def initial_state(self):
        return np.array([self.s0_, self.v0_])
    
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        s, v = x
        #v = max(v, 0.0)
        mu = self.r_ - self.q_ - 0.5 * v   #prime de risque 
        v_drift = self.kappa_ * (self.theta_ - v)
        return np.array([mu, v_drift])

    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        _, v = x
        #v = max(v, 0.0)
        vol = np.sqrt(v)
        sigma_v = self.sigma_ * vol
        sqrt_rho = np.sqrt(1.0 - self.rho_ * self.rho_)
        return np.array([[vol, 0.0], [self.rho_ * sigma_v, sqrt_rho * sigma_v]])

    
    # This is the continuous version of a characteristic function
    # M. Broadie, O. Kaya, Exact Simulation of Stochastic Volatility and other Affine Jump Diffusion Processes page8, formula 13 
    # http://finnp.stanford.edu/seminars/documents/Broadie.pdf
    # For details : Roger Lord, "Efficient Pricing Algorithms for exotic Derivatives"
    # http://repub.eur.nl/pub/13917/LordR-Thesis.pdf

    
    # Approximation du padé en analyse complexe est l'approximation d'une fonction analytique par une fraction rationnelle  
    # Pour approximer la transformée inverse de Fourier
    
    # Calcul l'integral ∫0_x ​sin(t)/t ​dt  méthode d'intégration numérique pour Broadie-Kaya qui approxime des intégrales dans le shéma exact de Heston 
    # Les coefficients sont tirés de la publication de C. Hastings Jr., "Approximations for Digital Computers", Princeton, 1955.
    # Page 103, Section 7.2 Sinus Integral Si(x)   a vérifier 
    #Cephes npematical Library (Stephen L. Moshier)
    