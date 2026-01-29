from schemes.explicit_euler import ExplicitEulerScheme
from schemes.implicit_euler import ImplicitEulerScheme


class CrankNicolsonScheme: 

    def __init__(self, theta, mapOperator, bc, tol, solverType):

        self.dt_ = None
        self.theta_ = theta
        self.explicit_ = ExplicitEulerScheme(mapOperator, bc)
        self.implicit_ = ImplicitEulerScheme(mapOperator, bc, tol, solverType)

    def step(self, u, t):
        if t - self.dt_ < -1e-8:
            raise ValueError("a step towards negative time is not allowed.")
        
        if self.theta_ != 1.0:
            self.explicit_.step(u, t, 1.0 - self.theta_)
        
        if self.theta_ != 0.0:
            self.implicit_.step(u, t, self.theta_)

    def setStep(self, dt):
        self.dt_ = dt
        self.explicit_.setStep(dt)
        self.implicit_.setStep(dt)  

    def numberOfIterations(self):
        return self.implicit_.numberOfIterations()